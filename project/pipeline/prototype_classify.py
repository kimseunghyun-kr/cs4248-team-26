"""
Phase 3 alternative: prototype-based evaluation for the official conditions.

Conditions:
  B1 (raw)              : raw encoder embeddings + raw class prompt prototypes
  D1 (debias_vl)        : debias_vl-projected embeddings + debiased class prompt prototypes
  D2 (CBDC)             : CBDC embeddings + CBDC class prompt prototypes
  D2.5 (CBDC no-label-select) : CBDC embeddings + label-free-selected CBDC prototypes
  D3 (debias_vl->CBDC)  : combined embeddings + combined class prompt prototypes
  D4 (adv-discovery->CBDC) : adversarial-discovery-fed CBDC embeddings + CBDC class prompt prototypes

No classifier head is trained. Predictions come from cosine similarity between
example embeddings and the condition-matched class prompt prototypes.

Run from project/ directory:
  python pipeline/prototype_classify.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from pipeline.artifacts import (
    condition_artifact_path,
    condition_dir,
    condition_split_path,
    ensure_condition_dir,
    iter_condition_labels,
    raw_split_path,
)


_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)
RESULTS_FILENAME = os.environ.get("RESULTS_FILE", "results_prototype.pt")

LABEL_NAMES = ["negative", "neutral", "positive"]
B1_LABEL = "B1 (raw)"


def _split_path(condition_label: str, split: str) -> str:
    if condition_label == B1_LABEL:
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


def load_condition_prototypes(condition_label: str) -> torch.Tensor | None:
    path = condition_artifact_path(CACHE_DIR, condition_label, "class_prompt_prototypes.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu").float()


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
def classify_with_prototypes(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z.float(), dim=-1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    return z @ prototypes.T


def main():
    parser = argparse.ArgumentParser(
        description="Prototype-based evaluation for B1 / D1 / D2 / D3."
    )
    _ = parser.parse_args()

    all_results = {}
    for condition_label in iter_condition_labels():
        print(f"\n{'=' * 72}")
        print(f"Condition: {condition_label}")
        print(f"{'=' * 72}")

        data = load_condition_data(condition_label)
        prototypes = load_condition_prototypes(condition_label)
        if data is None:
            print("  [skip] Required split embeddings not found")
            all_results[condition_label] = None
            continue
        if prototypes is None:
            print("  [skip] Required class_prompt_prototypes.pt not found")
            all_results[condition_label] = None
            continue

        ensure_condition_dir(CACHE_DIR, condition_label)

        val_logits = classify_with_prototypes(data["val"]["embeddings"], prototypes)
        test_logits = classify_with_prototypes(data["test"]["embeddings"], prototypes)

        val_acc, val_f1, _, _, _ = _gather_metrics(val_logits, data["val"]["labels"].long())
        test_acc, test_f1, report, disagreements, _ = _gather_metrics(
            test_logits,
            data["test"]["labels"].long(),
            texts=data["test"].get("texts"),
        )

        print(
            f"  val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"| test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
        )
        print("\n  Test classification report:")
        for line in report.strip().split("\n"):
            print(f"    {line}")

        if disagreements:
            csv_path = os.path.join(
                condition_dir(CACHE_DIR, condition_label),
                "prototype_test_disagreements.csv",
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["text", "true_label", "pred_label"])
                writer.writeheader()
                writer.writerows(disagreements)
            print(f"  Saved {len(disagreements)} disagreements -> {csv_path}")

        all_results[condition_label] = {
            "classifier_type": "prototype",
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "report": report,
            "artifact_dir": condition_dir(CACHE_DIR, condition_label),
        }

    print(f"\n{'=' * 72}")
    print("PROTOTYPE SUMMARY")
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

    out_path = os.path.join(CACHE_DIR, RESULTS_FILENAME)
    torch.save(all_results, out_path)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
