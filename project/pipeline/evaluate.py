"""
Phase 4: evaluation report for the supervised B1 / D1 / D2 / D3 pipeline.

Reads `results.pt`, summarizes supervised accuracy/F1, and reports
direction-interpretability diagnostics only when a condition-specific direction
artifact exists.

Run from project/ directory:
  python pipeline/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from pipeline.artifacts import condition_artifact_path, condition_split_path, iter_condition_labels


_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RESULTS_FILENAME = os.environ.get("RESULTS_FILE", "results.pt")
REPORT_FILENAME = os.environ.get("REPORT_FILE", "eval_report.txt")
EVAL_TITLE = os.environ.get("EVAL_TITLE", "CBDC Sentiment Debiasing — Supervised Evaluation Report")
RESULTS_SECTION_TITLE = os.environ.get("RESULTS_SECTION_TITLE", "Evaluation Results")


def direction_interpretability(z, labels, direction):
    """Mean projection score per sentiment class onto a condition-specific direction."""
    if direction.dim() == 2:
        direction = direction[0]
    d = F.normalize(direction, dim=-1)
    proj = (z @ d).numpy()
    labels_np = labels.numpy()

    result = {}
    for c, name in enumerate(["negative", "neutral", "positive"]):
        mask = labels_np == c
        result[name] = float(proj[mask].mean()) if mask.sum() > 0 else float("nan")
    result["pos_neg_gap"] = abs(result["positive"] - result["negative"])
    return result


def _load_condition_direction(condition_label: str) -> tuple[str, torch.Tensor] | None:
    filenames = [
        ("debias_vl_directions.pt", "debias_vl_directions"),
        ("cbdc_directions.pt", "cbdc_directions"),
    ]
    for filename, display_name in filenames:
        path = condition_artifact_path(CACHE_DIR, condition_label, filename)
        if os.path.exists(path):
            return display_name, torch.load(path, map_location="cpu")
    return None


def _load_condition_test_split(condition_label: str) -> dict | None:
    path = condition_split_path(CACHE_DIR, condition_label, "test")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lines = []

    def log(message: str = ""):
        print(message)
        lines.append(message)

    log("=" * 72)
    log(EVAL_TITLE)
    log("=" * 72)

    results_path = os.path.join(CACHE_DIR, RESULTS_FILENAME)
    if not os.path.exists(results_path):
        log(f"ERROR: {RESULTS_FILENAME} not found. Run the matching phase-3 classifier first.")
        return

    results = torch.load(results_path, map_location="cpu")

    log(f"\n--- {RESULTS_SECTION_TITLE} ---")
    log(f"{'Condition':<25} {'Val Acc':>8} {'Test Acc':>9} {'Val F1':>8} {'Test F1':>9}")
    log("-" * 68)
    for condition_label in iter_condition_labels():
        result = results.get(condition_label)
        if result is None:
            log(f"{condition_label:<25} {'N/A':>8} {'N/A':>9} {'N/A':>8} {'N/A':>9}")
            continue
        log(
            f"{condition_label:<25} "
            f"{result['val_accuracy']:>8.4f} {result['test_accuracy']:>9.4f} "
            f"{result['val_f1']:>8.4f} {result['test_f1']:>9.4f}"
        )

    log("\n--- Detailed Test Reports ---")
    for condition_label in iter_condition_labels():
        result = results.get(condition_label)
        if result is None:
            continue
        log(f"\n{condition_label}:")
        for line in result["report"].strip().split("\n"):
            log(f"  {line}")

    log("\n--- Direction Interpretability ---")
    log("(Lower pos-neg gap = direction is more sentiment-neutral)")

    header = f"{'Condition':<25} {'Direction':<20} {'neg_mean':>10} {'neu_mean':>10} {'pos_mean':>10} {'gap':>8}"
    log(header)
    log("-" * len(header))
    for condition_label in iter_condition_labels():
        direction_entry = _load_condition_direction(condition_label)
        test_data = _load_condition_test_split(condition_label)
        if direction_entry is None or test_data is None:
            continue

        direction_name, direction = direction_entry
        z_test = test_data["embeddings"]
        y_test = test_data["labels"]
        row = direction_interpretability(z_test, y_test, direction)
        log(
            f"{condition_label:<25} {direction_name:<20} "
            f"{row['negative']:>10.4f} {row['neutral']:>10.4f} "
            f"{row['positive']:>10.4f} {row['pos_neg_gap']:>8.4f}"
        )

    log("\n--- Comparative Analysis ---")

    def _metric(condition_label: str, key: str):
        result = results.get(condition_label)
        return None if result is None else result.get(key)

    b1_acc = _metric("B1 (raw)", "test_accuracy")
    d1_acc = _metric("D1 (debias_vl)", "test_accuracy")
    d2_acc = _metric("D2 (CBDC)", "test_accuracy")
    d3_acc = _metric("D3 (debias_vl->CBDC)", "test_accuracy")

    if b1_acc is not None and d1_acc is not None:
        delta = d1_acc - b1_acc
        log(f"  D1 vs B1 test accuracy: {delta:+.4f}")
    if b1_acc is not None and d2_acc is not None:
        delta = d2_acc - b1_acc
        log(f"  D2 vs B1 test accuracy: {delta:+.4f}")
    if b1_acc is not None and d3_acc is not None:
        delta = d3_acc - b1_acc
        log(f"  D3 vs B1 test accuracy: {delta:+.4f}")
    if d2_acc is not None and d3_acc is not None:
        delta = d3_acc - d2_acc
        log(f"  D3 vs D2 test accuracy: {delta:+.4f}")
    if d1_acc is not None and d3_acc is not None:
        delta = d3_acc - d1_acc
        log(f"  D3 vs D1 test accuracy: {delta:+.4f}")

    report_path = os.path.join(RESULTS_DIR, REPORT_FILENAME)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nFull report saved -> {report_path}")


if __name__ == "__main__":
    main()
