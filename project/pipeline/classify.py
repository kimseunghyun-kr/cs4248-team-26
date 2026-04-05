"""
Phase 2: classifier training and evaluation.

Modes:
  - linear: train a linear probe on cached Phase 1 embeddings
  - transformer: fine-tune a transformer classifier on cached tokens or
    reconstructed dataset records

Run from project/ directory:
  python pipeline/classify.py --classifier linear
  python pipeline/classify.py --classifier transformer
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import TransformerClassifierConfig
from dataset import (
    TextDataset,
    build_transformer_text_views,
    load_records,
    make_collate_fn,
    records_from_cached_payload,
)
from encoder import TransformerEncoder
from feature_utils import build_text_feature_matrix, get_requested_feature_names


LABEL_NAMES = ["negative", "neutral", "positive"]
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE = os.path.join(PROJECT_DIR, "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)
LINEAR_RESULTS_FILE = os.environ.get("LINEAR_RESULTS_FILE", "linear_results.pt")
LINEAR_CKPT_FILE = os.environ.get("LINEAR_CKPT_FILE", "linear_probe.pt")
TRANSFORMER_RESULTS_FILE = os.environ.get("TRANSFORMER_RESULTS_FILE", "transformer_results.pt")
TRANSFORMER_CKPT_FILE = os.environ.get("TRANSFORMER_CKPT_FILE", "transformer_classifier.pt")


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 768, n_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        return self.linear(x)


class CachedTokenDataset(Dataset):
    """Dataset view over Phase 1 cached token IDs and masks."""

    def __init__(self, payload: dict):
        self.input_ids = payload["input_ids"]
        self.attention_mask = payload["attention_mask"]
        self.labels = payload["labels"]
        self.texts = payload.get("texts") or [""] * len(self.labels)
        self.tweet_features = payload.get("tweet_features")
        if self.tweet_features is not None and not isinstance(self.tweet_features, torch.Tensor):
            self.tweet_features = torch.tensor(self.tweet_features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "text": self.texts[idx],
        }
        if self.tweet_features is not None:
            item["tweet_features"] = self.tweet_features[idx]
        return item


class FocalCrossEntropyLoss(nn.Module):
    """Focal cross-entropy for harder 3-way sentiment examples."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class ClassificationHead(nn.Module):
    """Classifier head with optional MLP projection."""

    def __init__(
        self,
        in_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        head_type: str = "mlp",
        hidden_dim: int = 0,
    ):
        super().__init__()
        if head_type not in {"linear", "mlp"}:
            raise ValueError("head_type must be 'linear' or 'mlp'")

        if hidden_dim <= 0:
            hidden_dim = in_dim

        layers = [nn.LayerNorm(in_dim)]
        if head_type == "linear":
            layers.extend([nn.Dropout(dropout), nn.Linear(in_dim, num_labels)])
        else:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_labels),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Decoder backbones like Qwen2/TinyLlama may emit bf16 activations while
        # the freshly initialized classifier head stays in fp32.
        first_param = next(self.net.parameters(), None)
        if first_param is not None:
            features = features.to(device=first_param.device, dtype=first_param.dtype)
        return self.net(features)


class TransformerClassifier(nn.Module):
    """Classifier head on top of the shared TransformerEncoder."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        num_labels: int = 3,
        dropout: float = 0.1,
        pooling: str = "cls",
        unfreeze_layers: int = 4,
        train_embeddings: bool = False,
        input_mode: str = "text_plus_selected",
        head_type: str = "mlp",
        hidden_dim: int = 0,
        tweet_feature_dim: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.input_mode = input_mode
        feature_dim = encoder.hidden_size * (4 if input_mode == "text_selected_pair" else 1)
        feature_dim += tweet_feature_dim
        self.classifier = ClassificationHead(
            in_dim=feature_dim,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
            hidden_dim=hidden_dim,
        )

        self.encoder.set_trainable_layers(
            n_layers=unfreeze_layers,
            train_embeddings=train_embeddings,
        )

    def _encode_view(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling=self.pooling,
            normalize=False,
        )

    def _fuse_pair_features(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [primary, secondary, torch.abs(primary - secondary), primary * secondary],
            dim=-1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        selected_input_ids: torch.Tensor | None = None,
        selected_attention_mask: torch.Tensor | None = None,
        tweet_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        primary = self._encode_view(input_ids, attention_mask)
        if self.input_mode == "text_selected_pair" and selected_input_ids is not None:
            secondary = self._encode_view(selected_input_ids, selected_attention_mask)
            features = self._fuse_pair_features(primary, secondary)
        else:
            features = primary
        if tweet_features is not None:
            tweet_features = tweet_features.to(device=features.device, dtype=features.dtype)
            features = torch.cat([features, tweet_features], dim=-1)
        return self.classifier(features)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def summarize_predictions(labels: list[int], preds: list[int]) -> dict:
    report_text = classification_report(
        labels,
        preds,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        labels,
        preds,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
        "report": report_text,
        "report_dict": report_dict,
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
    }


def train_probe(
    z_train,
    y_train,
    z_val,
    y_val,
    device,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
):
    in_dim = z_train.shape[1]
    model = LinearProbe(in_dim=in_dim, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(z_train, y_train), batch_size=batch_size, shuffle=True)

    best_val_f1 = 0.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for z_b, y_b in loader:
            z_b, y_b = z_b.to(device), y_b.to(device)
            opt.zero_grad()
            crit(model(z_b), y_b).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(z_val.to(device)).argmax(dim=-1).cpu().tolist()
        val_f1 = f1_score(y_val.tolist(), preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


def evaluate_probe(model, z, y, device) -> dict:
    model.eval()
    with torch.no_grad():
        preds = model(z.to(device)).argmax(dim=-1).cpu().tolist()
    return summarize_predictions(y.tolist(), preds)


def load_cached_split(split: str) -> dict | None:
    path = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def load_cached_embedding_splits() -> dict | None:
    payloads = {}
    for split in ["train", "val", "test"]:
        payload = load_cached_split(split)
        if payload is None:
            return None
        payloads[split] = payload
    return payloads


def text_features_requested(use_vader_features: bool, use_afinn_features: bool) -> bool:
    return use_vader_features or use_afinn_features


def get_payload_text_features(
    payload: dict,
    use_vader_features: bool = False,
    use_afinn_features: bool = False,
) -> torch.Tensor | None:
    requested_names = get_requested_feature_names(
        use_vader_features=use_vader_features,
        use_afinn_features=use_afinn_features,
    )
    if not requested_names:
        return None

    tweet_features = payload.get("tweet_features")
    cached_names = list(payload.get("tweet_feature_names") or [])
    if tweet_features is not None and cached_names == requested_names:
        if not isinstance(tweet_features, torch.Tensor):
            tweet_features = torch.tensor(tweet_features, dtype=torch.float32)
        return tweet_features.float()

    texts = payload.get("texts")
    if texts is None:
        raise ValueError("Requested text features, but cached payload has no texts to recompute from.")

    return build_text_feature_matrix(
        texts,
        use_vader_features=use_vader_features,
        use_afinn_features=use_afinn_features,
    )


def augment_embeddings_with_text_features(
    payload: dict,
    embeddings: torch.Tensor,
    use_vader_features: bool = False,
    use_afinn_features: bool = False,
) -> tuple[torch.Tensor, int]:
    tweet_features = get_payload_text_features(
        payload,
        use_vader_features=use_vader_features,
        use_afinn_features=use_afinn_features,
    )
    if tweet_features is None:
        return embeddings, 0
    return torch.cat([embeddings.float(), tweet_features.float()], dim=-1), int(tweet_features.shape[1])


def payload_has_requested_text_features(payload: dict, cfg: TransformerClassifierConfig) -> bool:
    requested_names = get_requested_feature_names(
        use_vader_features=cfg.use_vader_features,
        use_afinn_features=cfg.use_afinn_features,
    )
    if not requested_names:
        return True
    return (
        payload.get("tweet_features") is not None
        and list(payload.get("tweet_feature_names") or []) == requested_names
    )


def collate_cached_batch(batch: list[dict]) -> dict:
    collated = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "texts": [item["text"] for item in batch],
    }
    if "tweet_features" in batch[0]:
        collated["tweet_features"] = torch.stack(
            [item["tweet_features"].float() for item in batch],
            dim=0,
        )
    return collated


def resolve_transformer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_transformer_optimizer(
    model: TransformerClassifier,
    encoder_lr: float,
    classifier_lr: float,
    weight_decay: float,
):
    no_decay_terms = (
        "bias",
        "LayerNorm.weight",
        "LayerNorm.bias",
        "layer_norm.weight",
        "layer_norm.bias",
    )

    def group_named_params(named_params, lr: float):
        decay_params = []
        no_decay_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if any(term in name for term in no_decay_terms):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = []
        if decay_params:
            groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
        if no_decay_params:
            groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
        return groups

    param_groups = []
    param_groups.extend(group_named_params(model.encoder.backbone.named_parameters(), encoder_lr))
    param_groups.extend(group_named_params(model.classifier.named_parameters(), classifier_lr))

    if not param_groups:
        raise ValueError("No trainable parameters found for transformer optimizer.")

    return torch.optim.AdamW(param_groups)


def should_use_cached_tokens(cfg: TransformerClassifierConfig) -> bool:
    return (
        cfg.input_mode == "text"
        and not cfg.use_time_of_tweet
        and not cfg.use_age_of_user
        and not cfg.use_country
    )


def build_transformer_dataloaders(tokenizer, cfg: TransformerClassifierConfig) -> tuple[dict, str]:
    cached_payloads = {split: load_cached_split(split) for split in ["train", "val", "test"]}
    cached_feature_ready = all(
        payload is not None and payload_has_requested_text_features(payload, cfg)
        for payload in cached_payloads.values()
    )

    if (
        should_use_cached_tokens(cfg)
        and all(payload is not None for payload in cached_payloads.values())
        and cached_feature_ready
    ):
        print("Loading transformer classifier data from cached Phase 1 token splits.")
        datasets = {
            split: CachedTokenDataset(payload)
            for split, payload in cached_payloads.items()
        }
        collate_fn = collate_cached_batch
        source = "cache_tokens"
    else:
        if all(payload is not None for payload in cached_payloads.values()):
            print("Building transformer classifier views from cached Phase 1 records.")
            train_records = records_from_cached_payload(cached_payloads["train"])
            val_records = records_from_cached_payload(cached_payloads["val"])
            test_records = records_from_cached_payload(cached_payloads["test"])
            source = "cache_records"
        else:
            print("Cached Phase 1 splits missing; loading records from dataset.py.")
            train_records, val_records, test_records = load_records()
            source = "dataset_records"

        def build_dataset(records):
            texts, labels, secondary_texts = build_transformer_text_views(
                records,
                input_mode=cfg.input_mode,
                use_time_of_tweet=cfg.use_time_of_tweet,
                use_age_of_user=cfg.use_age_of_user,
                use_country=cfg.use_country,
            )
            tweet_features = build_text_feature_matrix(
                [record["text"] for record in records],
                use_vader_features=cfg.use_vader_features,
                use_afinn_features=cfg.use_afinn_features,
            )
            return TextDataset(
                texts,
                labels,
                tokenizer,
                max_length=cfg.max_length,
                secondary_texts=secondary_texts,
                tweet_features=tweet_features,
            )

        datasets = {
            "train": build_dataset(train_records),
            "val": build_dataset(val_records),
            "test": build_dataset(test_records),
        }
        collate_fn = make_collate_fn(tokenizer.pad_token_id or 0)

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
    }
    return loaders, source


def get_train_labels(loader) -> torch.Tensor:
    labels = loader.dataset.labels
    if isinstance(labels, torch.Tensor):
        return labels.cpu().long()
    return torch.tensor(labels, dtype=torch.long)


def build_transformer_loss(train_labels: torch.Tensor, cfg: TransformerClassifierConfig, device: str):
    class_weights = None
    if cfg.use_class_weights:
        counts = torch.bincount(train_labels, minlength=cfg.num_labels).float()
        class_weights = counts.sum() / counts.clamp_min(1.0)
        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.to(device)
        print(f"Class weights: {[round(x, 4) for x in class_weights.cpu().tolist()]}")

    if cfg.loss_name == "focal":
        criterion = FocalCrossEntropyLoss(
            weight=class_weights,
            gamma=cfg.focal_gamma,
            label_smoothing=cfg.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.label_smoothing,
        )
    return criterion, class_weights


def train_transformer_epoch(
    model: TransformerClassifier,
    loader,
    criterion,
    optimizer,
    scheduler,
    device: str,
    grad_clip_norm: float,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            selected_input_ids=batch.get("selected_input_ids").to(device) if "selected_input_ids" in batch else None,
            selected_attention_mask=batch.get("selected_attention_mask").to(device) if "selected_attention_mask" in batch else None,
            tweet_features=batch.get("tweet_features").to(device) if "tweet_features" in batch else None,
        )
        labels = batch["labels"].to(device)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = summarize_predictions(all_labels, all_preds)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def evaluate_transformer_classifier(
    model: TransformerClassifier,
    loader,
    criterion,
    device: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                selected_input_ids=batch.get("selected_input_ids").to(device) if "selected_input_ids" in batch else None,
                selected_attention_mask=batch.get("selected_attention_mask").to(device) if "selected_attention_mask" in batch else None,
                tweet_features=batch.get("tweet_features").to(device) if "tweet_features" in batch else None,
            )
            labels = batch["labels"].to(device)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = summarize_predictions(all_labels, all_preds)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def run_linear_probe_experiment(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    payloads = load_cached_embedding_splits()
    if payloads is None:
        raise FileNotFoundError(
            "Cached embeddings not found. Run data/embed.py first so z_tweet_{train,val,test}.pt exist."
        )

    z_train, text_feature_dim = augment_embeddings_with_text_features(
        payloads["train"],
        payloads["train"]["embeddings"],
        use_vader_features=args.use_vader_features,
        use_afinn_features=args.use_afinn_features,
    )
    y_train = payloads["train"]["labels"]
    z_val, _ = augment_embeddings_with_text_features(
        payloads["val"],
        payloads["val"]["embeddings"],
        use_vader_features=args.use_vader_features,
        use_afinn_features=args.use_afinn_features,
    )
    y_val = payloads["val"]["labels"]
    z_test, _ = augment_embeddings_with_text_features(
        payloads["test"],
        payloads["test"]["embeddings"],
        use_vader_features=args.use_vader_features,
        use_afinn_features=args.use_afinn_features,
    )
    y_test = payloads["test"]["labels"]

    print(f"Loaded cached embeddings: train={len(z_train)} val={len(z_val)} test={len(z_test)} "
          f"dim={z_train.shape[1]}")
    if text_feature_dim:
        enabled_names = get_requested_feature_names(
            use_vader_features=args.use_vader_features,
            use_afinn_features=args.use_afinn_features,
        )
        print(f"Augmented linear probe with text features: {enabled_names}")

    model, best_val_f1 = train_probe(z_train, y_train, z_val, y_val, device)
    train_metrics = evaluate_probe(model, z_train, y_train, device)
    val_metrics = evaluate_probe(model, z_val, y_val, device)
    test_metrics = evaluate_probe(model, z_test, y_test, device)

    print(f"Best val macro-F1: {best_val_f1:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f} | accuracy={test_metrics['accuracy']:.4f}")
    print("\nTest classification report:")
    for line in test_metrics["report"].strip().split("\n"):
        print(f"  {line}")

    ckpt_path = os.path.join(CACHE_DIR, LINEAR_CKPT_FILE)
    torch.save(model.state_dict(), ckpt_path)

    results = {
        "mode": "linear",
        "cache_dir": CACHE_DIR,
        "data_source": "cache_embeddings",
        "embedding_dim": int(z_train.shape[1]),
        "text_feature_dim": int(text_feature_dim),
        "text_feature_names": get_requested_feature_names(
            use_vader_features=args.use_vader_features,
            use_afinn_features=args.use_afinn_features,
        ),
        "best_val_f1": best_val_f1,
        "checkpoint_path": ckpt_path,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    results_path = os.path.join(CACHE_DIR, LINEAR_RESULTS_FILE)
    torch.save(results, results_path)
    print(f"\nLinear results saved -> {results_path}")
    print(f"Linear probe checkpoint -> {ckpt_path}")


def run_transformer_experiment(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = resolve_transformer_device()
    cfg = TransformerClassifierConfig(
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        n_epochs=args.n_epochs,
        encoder_lr=args.encoder_lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        unfreeze_layers=args.unfreeze_layers,
        train_embeddings=args.train_embeddings,
        pooling=args.pooling,
        input_mode=args.input_mode,
        use_time_of_tweet=args.use_time_of_tweet,
        use_age_of_user=args.use_age_of_user,
        use_country=args.use_country,
        use_vader_features=args.use_vader_features,
        use_afinn_features=args.use_afinn_features,
        loss_name=args.loss_name,
        focal_gamma=args.focal_gamma,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        use_class_weights=not args.no_class_weights,
        grad_clip_norm=args.grad_clip_norm,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Transformer model: {args.model_name}")
    print(f"Input mode: {cfg.input_mode}")
    print(f"Head: {cfg.head_type}")
    print(f"Loss: {cfg.loss_name}")
    if text_features_requested(cfg.use_vader_features, cfg.use_afinn_features):
        print(
            "Extra text features: "
            + ", ".join(
                get_requested_feature_names(
                    use_vader_features=cfg.use_vader_features,
                    use_afinn_features=cfg.use_afinn_features,
                )
            )
        )

    tokenizer = None
    tokenizer_name = os.environ.get("TOKENIZER_NAME")
    if tokenizer_name:
        print(f"Loading custom tokenizer from '{tokenizer_name}' ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encoder = TransformerEncoder(
        model_name=args.model_name,
        device=device,
        tokenizer=tokenizer,
    )

    loaders, data_source = build_transformer_dataloaders(encoder.tokenizer, cfg)
    print(f"Data source: {data_source}")
    print(
        f"Split sizes: train={len(loaders['train'].dataset)} | "
        f"val={len(loaders['val'].dataset)} | test={len(loaders['test'].dataset)}"
    )
    tweet_feature_dim = 0
    train_tweet_features = getattr(loaders["train"].dataset, "tweet_features", None)
    if isinstance(train_tweet_features, torch.Tensor):
        tweet_feature_dim = int(train_tweet_features.shape[1])
    expected_feature_names = get_requested_feature_names(
        use_vader_features=cfg.use_vader_features,
        use_afinn_features=cfg.use_afinn_features,
    )
    expected_feature_dim = len(expected_feature_names)
    if expected_feature_dim and tweet_feature_dim != expected_feature_dim:
        raise RuntimeError(
            "Requested text features "
            f"{expected_feature_names} (dim={expected_feature_dim}), but the phase-2 dataloader "
            f"provided dim={tweet_feature_dim}. This usually means the cached token payload or "
            "batch path did not carry tweet_features through correctly, or the run is mixing "
            "cache/config from a different feature setting. Use a fresh RUN_NAME and START_PHASE=1."
        )

    model = TransformerClassifier(
        encoder=encoder,
        num_labels=cfg.num_labels,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        unfreeze_layers=cfg.unfreeze_layers,
        train_embeddings=cfg.train_embeddings,
        input_mode=cfg.input_mode,
        head_type=cfg.head_type,
        hidden_dim=cfg.hidden_dim,
        tweet_feature_dim=tweet_feature_dim,
    ).to(device)
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")

    criterion, class_weights = build_transformer_loss(get_train_labels(loaders["train"]), cfg, device)
    optimizer = build_transformer_optimizer(
        model,
        encoder_lr=cfg.encoder_lr,
        classifier_lr=cfg.classifier_lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = max(1, len(loaders["train"]) * cfg.n_epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = float("-inf")
    patience_used = 0
    history = []
    ckpt_path = os.path.join(CACHE_DIR, TRANSFORMER_CKPT_FILE)

    for epoch in range(1, cfg.n_epochs + 1):
        train_metrics = train_transformer_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scheduler,
            device,
            cfg.grad_clip_norm,
        )
        val_metrics = evaluate_transformer_classifier(model, loaders["val"], criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        print(
            f"Epoch {epoch}/{cfg.n_epochs}: "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_used = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "best_epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "model_name": args.model_name,
                    "tokenizer_name": tokenizer_name,
                    "data_source": data_source,
                },
                ckpt_path,
            )
        else:
            patience_used += 1
            if patience_used >= cfg.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    results = {
        "mode": "transformer",
        "config": asdict(cfg),
        "model_name": args.model_name,
        "tokenizer_name": tokenizer_name,
        "cache_dir": CACHE_DIR,
        "data_source": data_source,
        "best_epoch": checkpoint["best_epoch"],
        "best_val_f1": checkpoint["best_val_f1"],
        "history": history,
        "class_weights": None if class_weights is None else class_weights.cpu().tolist(),
        "text_feature_dim": tweet_feature_dim,
        "text_feature_names": get_requested_feature_names(
            use_vader_features=cfg.use_vader_features,
            use_afinn_features=cfg.use_afinn_features,
        ),
    }

    for split in ["train", "val", "test"]:
        metrics = evaluate_transformer_classifier(model, loaders[split], criterion, device)
        results[split] = metrics
        print(
            f"Best checkpoint {split}: loss={metrics['loss']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f}"
        )

    results_path = os.path.join(CACHE_DIR, TRANSFORMER_RESULTS_FILE)
    torch.save(results, results_path)
    print(f"\nTransformer results saved -> {results_path}")
    print(f"Transformer checkpoint -> {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Classifier training and evaluation.")
    parser.add_argument("--classifier", choices=["linear", "transformer"], default="linear")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "bert-base-uncased"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze_layers", type=int, default=4)
    parser.add_argument("--pooling", default="auto")
    parser.add_argument(
        "--input_mode",
        choices=["text", "text_plus_selected", "text_selected_pair"],
        default="text_plus_selected",
    )
    parser.add_argument("--head_type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--loss_name", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--train_embeddings", action="store_true")
    parser.add_argument("--use_time_of_tweet", action="store_true")
    parser.add_argument("--use_age_of_user", action="store_true")
    parser.add_argument("--use_country", action="store_true")
    parser.add_argument("--use_vader_features", action="store_true")
    parser.add_argument("--use_afinn_features", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    args = parser.parse_args()

    if args.classifier == "transformer":
        run_transformer_experiment(args)
    else:
        run_linear_probe_experiment(args)


if __name__ == "__main__":
    main()
