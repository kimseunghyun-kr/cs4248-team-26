"""
Phase 4: Classification and evaluation.

Default behavior is unchanged:
  - train linear probes on cached embeddings for all experiment conditions

Optional transformer mode:
  - fine-tune a transformer classifier on the raw tokenized train/val/test
    splits cached by Phase 1 (`data/embed.py`)

Run from project/ directory:
  python pipeline/classify.py
  python pipeline/classify.py --classifier transformer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import TransformerClassifierConfig
from dataset import TextDataset, load_records, make_collate_fn
from encoder import TransformerEncoder
from pipeline.clean import materialize_sentiment_boost_conditions


LABEL_NAMES = ["negative", "neutral", "positive"]
_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)
TRANSFORMER_RESULTS_FILE = os.environ.get("TRANSFORMER_RESULTS_FILE", "transformer_results.pt")
TRANSFORMER_CKPT_FILE = os.environ.get("TRANSFORMER_CKPT_FILE", "transformer_classifier.pt")


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 768, n_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class CachedTokenDataset(Dataset):
    """Dataset view over Phase 1 cached token IDs / masks."""

    def __init__(self, payload: dict):
        self.input_ids = payload["input_ids"]
        self.attention_mask = payload["attention_mask"]
        self.labels = payload["labels"]
        self.texts = payload.get("texts") or [""] * len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "text": self.texts[idx],
        }


class TransformerClassifier(nn.Module):
    """Thin classifier head on top of the shared TransformerEncoder."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        num_labels: int = 3,
        dropout: float = 0.1,
        pooling: str = "cls",
        unfreeze_layers: int = 4,
        train_embeddings: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.hidden_size, num_labels)

        self.encoder.set_trainable_layers(
            n_layers=unfreeze_layers,
            train_embeddings=train_embeddings,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling=self.pooling,
            normalize=False,
        )
        return self.classifier(self.dropout(features))

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_probe(z_train, y_train, z_val, y_val, device,
                epochs=50, lr=1e-3, batch_size=256, weight_decay=1e-4):
    in_dim = z_train.shape[1]
    model = LinearProbe(in_dim=in_dim, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(z_train, y_train),
                        batch_size=batch_size, shuffle=True)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for z_b, y_b in loader:
            z_b, y_b = z_b.to(device), y_b.to(device)
            opt.zero_grad()
            crit(model(z_b), y_b).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(z_val.to(device)).argmax(dim=-1).cpu().numpy()
        val_f1 = f1_score(y_val.numpy(), preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


def evaluate_probe(model, z, y, device):
    model.eval()
    with torch.no_grad():
        preds = model(z.to(device)).argmax(dim=-1).cpu().numpy()
    labels_np = y.numpy()
    f1 = f1_score(labels_np, preds, average="macro")
    report = classification_report(
        labels_np, preds,
        target_names=LABEL_NAMES,
        digits=4,
    )
    return f1, report


def load_embeddings(name: str):
    results = {}
    for split in ["train", "val", "test"]:
        if name == "raw":
            path = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
        else:
            path = os.path.join(CACHE_DIR, f"z_tweet_{split}_{name}.pt")
        if not os.path.exists(path):
            return None
        results[split] = torch.load(path, map_location="cpu")
    return results


def _load_cached_raw_split(split: str) -> dict | None:
    path = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def collate_cached_batch(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "texts": [item["text"] for item in batch],
    }


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


def build_transformer_dataloaders(tokenizer, cfg: TransformerClassifierConfig) -> tuple[dict, str]:
    cached_payloads = {
        split: _load_cached_raw_split(split)
        for split in ["train", "val", "test"]
    }

    if all(payload is not None for payload in cached_payloads.values()):
        print("Loading transformer classifier data from cached Phase 1 splits.")
        datasets = {
            split: CachedTokenDataset(payload)
            for split, payload in cached_payloads.items()
        }
        collate_fn = collate_cached_batch
        source = "cache"
    else:
        print("Cached raw splits missing; falling back to dataset.py tokenization.")
        train_records, val_records, test_records = load_records()
        datasets = {
            "train": TextDataset(
                [r["text"] for r in train_records],
                [r["label"] for r in train_records],
                tokenizer,
                max_length=cfg.max_length,
            ),
            "val": TextDataset(
                [r["text"] for r in val_records],
                [r["label"] for r in val_records],
                tokenizer,
                max_length=cfg.max_length,
            ),
            "test": TextDataset(
                [r["text"] for r in test_records],
                [r["label"] for r in test_records],
                tokenizer,
                max_length=cfg.max_length,
            ),
        }
        collate_fn = make_collate_fn(tokenizer.pad_token_id or 0)
        source = "dataset"

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

    return {
        "loss": total_loss / max(1, len(loader)),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


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
            )
            labels = batch["labels"].to(device)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report_text = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": total_loss / max(1, len(loader)),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "accuracy": accuracy_score(all_labels, all_preds),
        "report": report_text,
        "report_dict": report_dict,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist(),
    }


# Condition name → cache suffix
CONDITIONS = {
    "B1 (raw)":            "raw",
    "D1 (debias_vl)":      "clean_debias_vl",
    "D2 (CBDC)":           "cbdc",
    "D3 (CBDC+proj)":      "clean_cbdc_proj",
    "D4 (raw+sent-boost)": "clean_raw_sentiment_boost",
    "D5 (CBDC+sent-boost)": "clean_cbdc_sentiment_boost",
    "C (label-guided)":    "clean_label_guided",
}

BOOST_SUFFIXES = {
    "clean_raw_sentiment_boost",
    "clean_cbdc_sentiment_boost",
}


def run_linear_probe_experiments():
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = {}

    for cond_name, cache_suffix in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        print(f"{'='*60}")

        data = load_embeddings(cache_suffix)
        if data is None and cache_suffix in BOOST_SUFFIXES:
            print("  missing boost embeddings; attempting to materialize them from Phase 2 artifacts ...")
            materialize_sentiment_boost_conditions(alpha=2.0)
            data = load_embeddings(cache_suffix)
        if data is None:
            print(f"  [skip] Embeddings not found for '{cache_suffix}'")
            all_results[cond_name] = None
            continue

        z_train = data["train"]["embeddings"]
        y_train = data["train"]["labels"]
        z_val   = data["val"]["embeddings"]
        y_val   = data["val"]["labels"]
        z_test  = data["test"]["embeddings"]
        y_test  = data["test"]["labels"]

        print(f"  train={len(z_train)} val={len(z_val)} test={len(z_test)} dim={z_train.shape[1]}")

        model, best_val_f1 = train_probe(z_train, y_train, z_val, y_val, device)
        test_f1, report = evaluate_probe(model, z_test, y_test, device)

        print(f"  val_f1={best_val_f1:.4f} | test_f1={test_f1:.4f}")
        print(f"\n  Test classification report:")
        for line in report.strip().split("\n"):
            print(f"    {line}")

        all_results[cond_name] = {
            "val_f1":  best_val_f1,
            "test_f1": test_f1,
            "report":  report,
        }

        probe_path = os.path.join(CACHE_DIR, f"probe_{cache_suffix}.pt")
        torch.save(model.state_dict(), probe_path)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'Val F1':>8} {'Test F1':>9}")
    print("-" * 45)
    for cond_name, res in all_results.items():
        if res is None:
            print(f"{cond_name:<25} {'N/A':>8} {'N/A':>9}")
        else:
            print(f"{cond_name:<25} {res['val_f1']:>8.4f} {res['test_f1']:>9.4f}")

    out_path = os.path.join(CACHE_DIR, "results.pt")
    torch.save(all_results, out_path)
    print(f"\nResults saved -> {out_path}")


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
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        use_class_weights=not args.no_class_weights,
        grad_clip_norm=args.grad_clip_norm,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Transformer model: {args.model_name}")

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
    print(f"Split sizes: train={len(loaders['train'].dataset)} | "
          f"val={len(loaders['val'].dataset)} | test={len(loaders['test'].dataset)}")

    model = TransformerClassifier(
        encoder=encoder,
        num_labels=cfg.num_labels,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        unfreeze_layers=cfg.unfreeze_layers,
        train_embeddings=cfg.train_embeddings,
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
    best_epoch = 0
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

        print(f"Epoch {epoch}/{cfg.n_epochs}: "
              f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_used = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "best_epoch": best_epoch,
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
    }

    for split in ["train", "val", "test"]:
        metrics = evaluate_transformer_classifier(model, loaders[split], criterion, device)
        results[split] = metrics
        print(f"Best checkpoint {split}: loss={metrics['loss']:.4f} "
              f"macro_f1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f}")

    results_path = os.path.join(CACHE_DIR, TRANSFORMER_RESULTS_FILE)
    torch.save(results, results_path)
    print(f"\nTransformer results saved -> {results_path}")
    print(f"Transformer checkpoint -> {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Classifier training + evaluation.")
    parser.add_argument("--classifier", choices=["linear", "transformer"], default="linear",
                        help="Classifier mode. 'linear' preserves the original Phase 4 behavior.")
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
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--train_embeddings", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    args = parser.parse_args()

    if args.classifier == "transformer":
        run_transformer_experiment(args)
    else:
        run_linear_probe_experiments()


if __name__ == "__main__":
    main()
