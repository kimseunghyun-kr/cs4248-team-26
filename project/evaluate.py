"""
Evaluation: load checkpoint, run test set, produce reports.

Reports:
  1. Classification report
  2. Confusion matrix
  3. Direction interpretability table (mean projection per sentiment class)
  4. Ablation comparison
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model import CBDCSentimentClassifier
from encoder import FinBERTEncoder
from direction_bank import FinancialDirectionBank
from dataset import TweetDataset, make_collate_fn
from config import TrainConfig
from train import evaluate as eval_loop


LABEL_NAMES = ["negative", "neutral", "positive"]


# ---------------------------------------------------------------------------
# Load a trained model from checkpoint
# ---------------------------------------------------------------------------
def load_model(
    ckpt_path: str,
    encoder: FinBERTEncoder,
    bank: FinancialDirectionBank,
    cfg: TrainConfig,
    ablation: str = "full",
) -> CBDCSentimentClassifier:
    model = CBDCSentimentClassifier(encoder, bank, cfg, ablation=ablation)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.to(cfg.device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Full evaluation on test set
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_test_evaluation(
    model: CBDCSentimentClassifier,
    test_loader: DataLoader,
    device: str,
) -> dict:
    all_preds, all_labels, all_texts = [], [], []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_texts.extend(batch["texts"])

    report = classification_report(
        all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds)
    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "preds": all_preds,
        "labels": all_labels,
        "texts": all_texts,
    }


# ---------------------------------------------------------------------------
# Direction interpretability table
# ---------------------------------------------------------------------------
@torch.no_grad()
def direction_interpretability(
    model: CBDCSentimentClassifier,
    test_loader: DataLoader,
    device: str,
) -> dict:
    """
    For each axis, compute mean projection score grouped by true label.
    Returns dict: axis_name → {0: mean, 1: mean, 2: mean}
    """
    bank = model.bank
    encoder = model.encoder

    all_proj = {name: [] for name in bank.axis_names}
    all_labels = []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        z = encoder.encode_ids(input_ids, attention_mask)  # B × H
        proj = bank.get_feature_vector(z)                   # B × K

        for i, name in enumerate(bank.axis_names):
            all_proj[name].extend(proj[:, i].cpu().tolist())
        all_labels.extend(labels.tolist())

    all_labels = np.array(all_labels)
    table = {}
    for name in bank.axis_names:
        vals = np.array(all_proj[name])
        table[name] = {
            label: float(vals[all_labels == lbl_idx].mean())
            for lbl_idx, label in enumerate(LABEL_NAMES)
        }
    return table


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def print_classification_report(report: dict):
    print("\n  Classification Report:")
    header = f"  {'':15s} {'precision':>10s} {'recall':>8s} {'f1-score':>10s} {'support':>8s}"
    print(header)
    for label in LABEL_NAMES + ["macro avg", "weighted avg"]:
        if label in report:
            r = report[label]
            print(
                f"  {label:15s} "
                f"{r['precision']:>10.4f} "
                f"{r['recall']:>8.4f} "
                f"{r['f1-score']:>10.4f} "
                f"{int(r.get('support', 0)):>8d}"
            )


def print_confusion_matrix(cm: list):
    print("\n  Confusion Matrix:")
    print(f"  {'':10s}", end="")
    for name in LABEL_NAMES:
        print(f"  {name[:7]:>7s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:10s}", end="")
        for val in row:
            print(f"  {val:>7d}", end="")
        print()


def print_interpretability_table(table: dict):
    print("\n  Direction Interpretability:")
    print(f"  {'axis':22s}  {'neg_mean':>9s}  {'neu_mean':>9s}  {'pos_mean':>9s}")
    print("  " + "-" * 54)
    for axis, vals in table.items():
        neg = vals["negative"]
        neu = vals["neutral"]
        pos = vals["positive"]
        print(f"  {axis:22s}  {neg:>9.4f}  {neu:>9.4f}  {pos:>9.4f}")


def print_ablation(ablation_results: dict):
    print("\n  Ablation Results:")
    order = ["z_only", "directions_only", "full"]
    label_map = {
        "z_only": "z only",
        "directions_only": "directions only",
        "full": "z + directions",
    }
    for key in order:
        if key in ablation_results:
            f1 = ablation_results[key]
            tag = "  ← full model" if key == "full" else ""
            print(f"  {label_map[key]:20s}  val_f1 = {f1:.4f}{tag}")


# ---------------------------------------------------------------------------
# Save full report to file
# ---------------------------------------------------------------------------
def save_report(
    results_dir: str,
    test_results: dict,
    interp_table: dict,
    ablation_results: dict,
):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "eval_report.txt")
    import io
    import sys

    # Capture printed output
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print("=" * 60)
    print("CBDC Financial Sentiment Classifier — Evaluation Report")
    print("=" * 60)
    print_classification_report(test_results["report"])
    print_confusion_matrix(test_results["confusion_matrix"])
    print_interpretability_table(interp_table)
    print_ablation(ablation_results)
    print("=" * 60)

    sys.stdout = old_stdout
    content = buf.getvalue()
    print(content)

    with open(path, "w") as f:
        f.write(content)
    print(f"Report saved to {path}")
