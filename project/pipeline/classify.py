"""
Phase 4: Train and evaluate linear probes for all experiment conditions.

Conditions:
  B1 (raw)          : raw BERT/FinBERT CLS embeddings (baseline)
  D1 (debias_vl)    : debias_vl word-pair projection applied to raw embeddings
  D2 (CBDC)         : CBDC text_iccv fine-tuned encoder embeddings
  D3 (CBDC+proj)    : CBDC encoder + residual CBDC direction projection
  D4 (debias+boost) : CBDC confound removal + sentiment-subspace amplification (raw)
  D5 (CBDC+boost)   : CBDC confound removal + sentiment-subspace amplification (CBDC)
  C (label-guided)  : label-guided within-class mean-shift projection (oracle)

All conditions train a Linear(768, 3) probe and evaluate with macro F1.

Run from project/ directory:
  python pipeline/classify.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
import numpy as np

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 768, n_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


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
        target_names=["negative", "neutral", "positive"],
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


# Condition name → cache suffix
CONDITIONS = {
    "B1 (raw)":            "raw",
    "D1 (debias_vl)":      "clean_debias_vl",
    "D2 (CBDC)":           "cbdc",
    "D3 (CBDC+proj)":      "clean_cbdc_proj",
    "D4 (debias+boost)":   "clean_debias_vl_boost",
    "D5 (CBDC+boost)":     "clean_cbdc_boost",
    "C (label-guided)":    "clean_label_guided",
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = {}

    for cond_name, cache_suffix in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        print(f"{'='*60}")

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

    # Summary
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


if __name__ == "__main__":
    main()
