"""
Phase 3: evaluation reporting for sentiment-classification experiments.

Modes:
  - linear: report the linear-probe baseline
  - transformer: report the fine-tuned transformer classifier
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE = os.path.join(PROJECT_DIR, "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
LINEAR_RESULTS_FILE = os.environ.get("LINEAR_RESULTS_FILE", "linear_results.pt")
TRANSFORMER_RESULTS_FILE = os.environ.get("TRANSFORMER_RESULTS_FILE", "transformer_results.pt")
LEGACY_LINEAR_RESULTS_FILE = "results.pt"


def _load_results(results_file: str) -> dict:
    path = os.path.join(CACHE_DIR, results_file)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def _extract_linear_baseline(results: dict) -> dict | None:
    if results.get("mode") == "linear" and "test" in results:
        return results
    raw = results.get("B1 (raw)")
    if raw is None:
        return None
    return {
        "mode": "linear-legacy",
        "cache_dir": CACHE_DIR,
        "best_val_f1": raw.get("val_f1"),
        "test": {
            "macro_f1": raw.get("test_f1"),
            "report": raw.get("report", ""),
        },
    }


def run_linear_evaluation():
    lines = []

    def log(s: str = ""):
        print(s)
        lines.append(s)

    try:
        results = _load_results(LINEAR_RESULTS_FILE)
    except FileNotFoundError:
        try:
            results = _load_results(LEGACY_LINEAR_RESULTS_FILE)
        except FileNotFoundError:
            log(f"ERROR: {LINEAR_RESULTS_FILE} not found. Run classify.py --classifier linear first.")
            return

    baseline = _extract_linear_baseline(results)
    if baseline is None:
        log("ERROR: Unable to interpret linear-probe results format.")
        return

    log("=" * 70)
    log("Linear Probe Sentiment Baseline — Evaluation Report")
    log("=" * 70)
    log(f"Cache dir: {baseline.get('cache_dir', CACHE_DIR)}")
    if baseline.get("best_val_f1") is not None:
        log(f"Best val macro-F1: {baseline['best_val_f1']:.4f}")

    if baseline.get("mode") == "linear":
        log(f"Checkpoint: {baseline.get('checkpoint_path')}")
        log("\n--- Split Summary ---")
        log(f"{'Split':<10} {'Macro F1':>10} {'Accuracy':>10}")
        log("-" * 34)
        for split in ["train", "val", "test"]:
            metrics = baseline[split]
            log(
                f"{split:<10} {metrics['macro_f1']:>10.4f} "
                f"{metrics['accuracy']:>10.4f}"
            )
    else:
        log("\nLegacy baseline results detected.")
        log(f"Test macro-F1: {baseline['test'].get('macro_f1', float('nan')):.4f}")

    report = baseline["test"].get("report", "").strip()
    if report:
        log("\n--- TEST Classification Report ---")
        for line in report.split("\n"):
            log(line)

    confusion = baseline["test"].get("confusion_matrix")
    if confusion:
        log("\nTEST confusion matrix:")
        for row in confusion:
            log("  " + " ".join(f"{int(v):>5d}" for v in row))

    report_path = os.path.join(RESULTS_DIR, "linear_eval_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nFull report saved -> {report_path}")


def run_transformer_evaluation():
    lines = []

    def log(s: str = ""):
        print(s)
        lines.append(s)

    results_path = os.path.join(CACHE_DIR, TRANSFORMER_RESULTS_FILE)
    if not os.path.exists(results_path):
        log(f"ERROR: {TRANSFORMER_RESULTS_FILE} not found. Run classify.py --classifier transformer first.")
        return

    results = torch.load(results_path, map_location="cpu")

    log("=" * 70)
    log("Transformer Sentiment Classifier — Evaluation Report")
    log("=" * 70)
    log(f"Model: {results.get('model_name')}")
    log(f"Cache dir: {results.get('cache_dir', CACHE_DIR)}")
    log(f"Data source: {results.get('data_source', 'unknown')}")
    log(f"Best epoch: {results.get('best_epoch')} | best val_f1={results.get('best_val_f1', float('nan')):.4f}")

    log("\n--- Split Summary ---")
    log(f"{'Split':<10} {'Loss':>8} {'Macro F1':>10} {'Accuracy':>10}")
    log("-" * 42)
    for split in ["train", "val", "test"]:
        split_metrics = results[split]
        log(
            f"{split:<10} {split_metrics['loss']:>8.4f} "
            f"{split_metrics['macro_f1']:>10.4f} {split_metrics['accuracy']:>10.4f}"
        )

    for split in ["val", "test"]:
        split_metrics = results[split]
        log(f"\n--- {split.upper()} Classification Report ---")
        for line in split_metrics["report"].strip().split("\n"):
            log(line)

        log(f"\n{split.upper()} confusion matrix:")
        for row in split_metrics["confusion_matrix"]:
            log("  " + " ".join(f"{int(v):>5d}" for v in row))

    baseline = None
    for filename in [LINEAR_RESULTS_FILE, LEGACY_LINEAR_RESULTS_FILE]:
        path = os.path.join(CACHE_DIR, filename)
        if os.path.exists(path):
            baseline = _extract_linear_baseline(torch.load(path, map_location="cpu"))
            if baseline is not None:
                break

    if baseline is not None and baseline.get("test", {}).get("macro_f1") is not None:
        delta = results["test"]["macro_f1"] - baseline["test"]["macro_f1"]
        log("\n--- Comparison To Linear Probe ---")
        log(f"Transformer test F1: {results['test']['macro_f1']:.4f}")
        log(f"Linear probe test F1: {baseline['test']['macro_f1']:.4f}")
        log(f"Delta: {delta:+.4f}")

    report_path = os.path.join(RESULTS_DIR, "transformer_eval_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nFull report saved -> {report_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Evaluation reporting.")
    parser.add_argument("--mode", choices=["linear", "transformer"], default="linear")
    args = parser.parse_args()

    if args.mode == "transformer":
        run_transformer_evaluation()
    else:
        run_linear_evaluation()


if __name__ == "__main__":
    main()
