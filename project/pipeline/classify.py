"""
Phase 5: Train and evaluate a linear probe for all experiment conditions.

Conditions:
  B1          : raw FinBERT CLS embeddings (baseline)
  B2          : SAE-projected embeddings   (v_style projected out)
  B2.5        : mean-shift projected       (v_shift projected out)
  B3          : SAE + CBDC projected       (delta_star projected out) ← main method
  B4          : trained linear probe on raw z (upper bound, same as B1 but explicit)
  C           : label-guided projected     (label_guided direction projected out)

All conditions train a Linear(768, 3) probe on the respective embeddings and
evaluate on the test split using macro F1.

Outputs saved to cache/results.pt:
  Dict mapping condition name → {"val_f1": float, "test_f1": float, "report": str}

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

# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 768, n_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val:   torch.Tensor,
    y_val:   torch.Tensor,
    device:  str,
    epochs:  int = 50,
    lr:      float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
) -> tuple[LinearProbe, float]:
    """
    Train a linear probe and return (model, best_val_f1).
    Uses early stopping on val F1.
    """
    in_dim = z_train.shape[1]
    model  = LinearProbe(in_dim=in_dim, n_classes=3).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit   = nn.CrossEntropyLoss()

    dataset = TensorDataset(z_train, y_train)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_f1  = 0.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        for z_b, y_b in loader:
            z_b, y_b = z_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(model(z_b), y_b)
            loss.backward()
            opt.step()

        # Validate
        model.eval()
        with torch.no_grad():
            logits = model(z_val.to(device))
            preds  = logits.argmax(dim=-1).cpu().numpy()
        val_f1 = f1_score(y_val.numpy(), preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


def evaluate_probe(
    model: LinearProbe,
    z: torch.Tensor,
    y: torch.Tensor,
    device: str,
) -> tuple[float, str]:
    """Returns (macro_f1, classification_report_str)."""
    model.eval()
    with torch.no_grad():
        logits = model(z.to(device))
        preds  = logits.argmax(dim=-1).cpu().numpy()
    labels_np = y.numpy()
    f1     = f1_score(labels_np, preds, average="macro")
    report = classification_report(
        labels_np, preds,
        target_names=["negative", "neutral", "positive"],
        digits=4,
    )
    return f1, report


# ---------------------------------------------------------------------------
# Load embeddings helper
# ---------------------------------------------------------------------------
def load_embeddings(name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load (embeddings, labels) from cache for a given condition name.
    name examples: "raw", "clean_delta_star", "clean_v_style", ...
    Returns None if files missing.
    """
    results = {}
    for split in ["train", "val", "test"]:
        if name == "raw":
            path = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
        else:
            path = os.path.join(CACHE_DIR, f"z_tweet_{split}_{name}.pt")

        if not os.path.exists(path):
            return None
        data = torch.load(path, map_location="cpu")
        results[split] = data

    return results


# ---------------------------------------------------------------------------
# Run all conditions
# ---------------------------------------------------------------------------
CONDITIONS = {
    # condition_name : cache_suffix (maps to z_tweet_{split}_{suffix}.pt or z_tweet_{split}.pt)
    "B1 (raw)"            : "raw",
    "B2 (SAE)"            : "clean_v_style",
    "B2.5 (mean-shift)"   : "clean_v_shift",
    "B3 (SAE+CBDC)"       : "clean_delta_star",
    "C (label-guided)"    : "clean_label_guided",
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
            print(f"  ⚠ Embeddings not found for suffix '{cache_suffix}', skipping.")
            all_results[cond_name] = None
            continue

        z_train = data["train"]["embeddings"]
        y_train = data["train"]["labels"]
        z_val   = data["val"]["embeddings"]
        y_val   = data["val"]["labels"]
        z_test  = data["test"]["embeddings"]
        y_test  = data["test"]["labels"]

        print(f"  train={len(z_train)} val={len(z_val)} test={len(z_test)} dim={z_train.shape[1]}")

        model, best_val_f1 = train_probe(
            z_train, y_train, z_val, y_val,
            device=device, epochs=50, lr=1e-3, batch_size=256,
        )

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

        # Save probe checkpoint
        probe_path = os.path.join(CACHE_DIR, f"probe_{cache_suffix}.pt")
        torch.save(model.state_dict(), probe_path)

    # ---- Summary table -------------------------------------------------------
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

    # ---- Save all results ---------------------------------------------------
    out_path = os.path.join(CACHE_DIR, "results.pt")
    torch.save(all_results, out_path)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
