"""
Phase 3: supervised linear-probe evaluation for the official conditions.

Conditions:
  B1 (raw)              : raw encoder embeddings
  D1 (debias_vl)        : debias_vl-projected embeddings
  D2 (CBDC)             : pure prompt-driven CBDC encoder embeddings
  D3 (debias_vl->CBDC)  : combined debias_vl-fed CBDC encoder embeddings
  D4 (adv-discovery->CBDC) : adversarial-discovery-fed CBDC encoder embeddings

Each condition trains a lightweight linear probe on train embeddings,
selects by validation accuracy, and reports accuracy + macro F1 on val/test.

Run from project/ directory:
  python pipeline/classify.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from pipeline.artifacts import (
    condition_dir,
    condition_split_path,
    ensure_condition_dir,
    iter_condition_labels,
    raw_split_path,
)


_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)

LABEL_NAMES = ["negative", "neutral", "positive"]


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_path(condition_label: str, split: str) -> str:
    if condition_label == "B1 (raw)":
        return raw_split_path(CACHE_DIR, split)
    return condition_split_path(CACHE_DIR, condition_label, split)


def load_condition_data(condition_label: str) -> dict | None:
    payload = {}
    for split in ["train", "val", "test"]:
        path = _split_path(condition_label, split)
        if not os.path.exists(path):
            return None
        payload[split] = torch.load(path, map_location="cpu")
    return payload


def _gather_metrics(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    texts: list[str] | None = None,
) -> tuple[float, float, str, list[dict], list[int]]:
    preds = logits.argmax(dim=-1).cpu().tolist()
    labels = y_true.cpu().tolist()

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    report = classification_report(
        labels,
        preds,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    disagreements = []
    if texts is not None:
        for idx, (pred, gold) in enumerate(zip(preds, labels)):
            if pred == gold:
                continue
            disagreements.append(
                {
                    "text": texts[idx],
                    "true_label": LABEL_NAMES[gold],
                    "pred_label": LABEL_NAMES[pred],
                }
            )

    return accuracy, macro_f1, report, disagreements, preds


@torch.no_grad()
def _forward(model: nn.Module, z: torch.Tensor, device: str) -> torch.Tensor:
    return model(z.to(device)).cpu()


def train_linear_probe(
    train_data: dict,
    val_data: dict,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
) -> tuple[nn.Module, dict]:
    z_train = train_data["embeddings"].float()
    y_train = train_data["labels"].long()
    z_val = val_data["embeddings"].float()
    y_val = val_data["labels"].long()

    model = LinearProbe(z_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_metrics = None
    best_key = (-1.0, -1.0)
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(z_train))
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            batch_x = z_train[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_logits = _forward(model, z_val, device)
        val_acc, val_f1, _, _, _ = _gather_metrics(val_logits, y_val)
        current_key = (val_acc, val_f1)
        if current_key > best_key:
            best_key = current_key
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_metrics = {
                "best_epoch": epoch + 1,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Linear probe training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    return model, best_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Supervised linear-probe evaluation for B1 / D1 / D2 / D3 / D4."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = {}
    for condition_label in iter_condition_labels():
        print(f"\n{'=' * 72}")
        print(f"Condition: {condition_label}")
        print(f"{'=' * 72}")

        data = load_condition_data(condition_label)
        if data is None:
            print("  [skip] Required split embeddings not found")
            all_results[condition_label] = None
            continue

        ensure_condition_dir(CACHE_DIR, condition_label)
        probe, best_metrics = train_linear_probe(
            train_data=data["train"],
            val_data=data["val"],
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
        )

        probe.eval()
        val_logits = _forward(probe, data["val"]["embeddings"].float(), device)
        test_logits = _forward(probe, data["test"]["embeddings"].float(), device)

        val_acc, val_f1, _, _, _ = _gather_metrics(val_logits, data["val"]["labels"].long())
        test_acc, test_f1, report, disagreements, _ = _gather_metrics(
            test_logits,
            data["test"]["labels"].long(),
            texts=data["test"].get("texts"),
        )

        print(
            f"  best_epoch={best_metrics['best_epoch']} "
            f"| val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"| test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
        )
        print("\n  Test classification report:")
        for line in report.strip().split("\n"):
            print(f"    {line}")

        if disagreements:
            csv_path = os.path.join(condition_dir(CACHE_DIR, condition_label), "test_disagreements.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["text", "true_label", "pred_label"])
                writer.writeheader()
                writer.writerows(disagreements)
            print(f"  Saved {len(disagreements)} disagreements -> {csv_path}")

        all_results[condition_label] = {
            "best_epoch": best_metrics["best_epoch"],
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "report": report,
            "artifact_dir": condition_dir(CACHE_DIR, condition_label),
        }

    print(f"\n{'=' * 72}")
    print("SUPERVISED SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'Condition':<25} {'Val Acc':>8} {'Test Acc':>9} {'Val F1':>8} {'Test F1':>9}")
    print("-" * 68)
    for condition_label, result in all_results.items():
        if result is None:
            print(f"{condition_label:<25} {'N/A':>8} {'N/A':>9} {'N/A':>8} {'N/A':>9}")
            continue
        print(
            f"{condition_label:<25} "
            f"{result['val_accuracy']:>8.4f} {result['test_accuracy']:>9.4f} "
            f"{result['val_f1']:>8.4f} {result['test_f1']:>9.4f}"
        )

    out_path = os.path.join(CACHE_DIR, "results.pt")
    torch.save(all_results, out_path)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
