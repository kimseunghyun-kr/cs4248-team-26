"""
Phase 6: Full evaluation report — direction interpretability + zero-shot check.

Loads the saved results from classify.py and produces:
  1. Summary table (all conditions: val F1, test F1)
  2. Direction interpretability — mean projection score per sentiment class
     for each direction (delta_star, v_style, v_shift, label_guided)
  3. Style axis linearity check — cosine(z + α*d, formal_centroid) vs α
  4. Zero-shot preservation check on formal corpus (Financial PhraseBank)
  5. Writes full report to results/eval_report_new.txt

Run from project/ directory:
  python pipeline/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, classification_report

from pipeline.classify import LinearProbe, CONDITIONS

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR   = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# Direction interpretability
# ---------------------------------------------------------------------------
def direction_interpretability(
    z: torch.Tensor,           # (N, 768)
    labels: torch.Tensor,      # (N,)
    direction: torch.Tensor,   # (768,)
    direction_name: str,
) -> dict:
    """
    For each sentiment class, compute mean projection score onto direction.
    A style-only direction should show similar scores across classes.
    A sentiment-correlated direction would show large differences.
    """
    d = F.normalize(direction, dim=-1)
    projections = (z @ d).numpy()    # (N,) scalar projections
    labels_np   = labels.numpy()

    result = {}
    for c, c_name in enumerate(["negative", "neutral", "positive"]):
        mask = labels_np == c
        if mask.sum() > 0:
            result[c_name] = float(projections[mask].mean())
        else:
            result[c_name] = float("nan")

    gap = abs(result["positive"] - result["negative"])
    result["pos_neg_gap"] = gap
    return result


def print_interpretability_table(interp_results: dict) -> str:
    """Format and return the interpretability table as a string."""
    lines = []
    header = f"{'Direction':<20} {'neg_mean':>10} {'neu_mean':>10} {'pos_mean':>10} {'pos-neg gap':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for d_name, row in interp_results.items():
        lines.append(
            f"{d_name:<20} "
            f"{row.get('negative', float('nan')):>10.4f} "
            f"{row.get('neutral',  float('nan')):>10.4f} "
            f"{row.get('positive', float('nan')):>10.4f} "
            f"{row.get('pos_neg_gap', float('nan')):>12.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Style axis linearity check
# ---------------------------------------------------------------------------
def linearity_check(
    z_sample: torch.Tensor,    # (20, 768) sample tweets
    direction: torch.Tensor,   # (768,)
    formal_centroid: torch.Tensor,  # (768,)
    alphas: list = None,
) -> dict:
    """
    For α in alphas, compute mean cosine(z + α*d, formal_centroid).
    If direction is a genuine style axis, this should be monotonic.
    """
    if alphas is None:
        alphas = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

    d  = F.normalize(direction, dim=-1)
    fc = F.normalize(formal_centroid, dim=-1)

    results = {}
    for alpha in alphas:
        z_shifted = F.normalize(z_sample + alpha * d.unsqueeze(0), dim=-1)
        cos       = F.cosine_similarity(z_shifted, fc.unsqueeze(0), dim=-1).mean().item()
        results[alpha] = cos
    return results


# ---------------------------------------------------------------------------
# Zero-shot preservation check
# ---------------------------------------------------------------------------
def zero_shot_preservation_check(
    z_formal: torch.Tensor,    # (M, 768) formal embeddings
    direction: torch.Tensor,   # (768,) direction to project out
    label_map: dict = None,
) -> dict:
    """
    Applies projection to formal embeddings and checks cosine similarity
    with original (proxy for zero-shot preservation).
    High mean cosine = direction removed little from formal = preserved.
    """
    from pipeline.clean import project_out
    z_clean = project_out(z_formal, direction)

    # Cosine similarity between cleaned and original (per-sample)
    cos_per_sample = F.cosine_similarity(z_clean, z_formal, dim=-1)  # (M,)
    return {
        "mean_cosine_preserved": cos_per_sample.mean().item(),
        "min_cosine_preserved":  cos_per_sample.min().item(),
        "std_cosine_preserved":  cos_per_sample.std().item(),
    }


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("CBDC Financial Sentiment Classifier — Full Evaluation Report (New)")
    log("=" * 70)

    # ---- Load results from classify.py --------------------------------------
    results_path = os.path.join(CACHE_DIR, "results.pt")
    if not os.path.exists(results_path):
        log("ERROR: results.pt not found. Run `python pipeline/classify.py` first.")
        return

    results = torch.load(results_path, map_location="cpu")

    log("\n--- Classification Results ---")
    log(f"{'Condition':<25} {'Val F1':>8} {'Test F1':>9}")
    log("-" * 45)
    for cond, res in results.items():
        if res is None:
            log(f"{cond:<25} {'N/A':>8} {'N/A':>9}")
        else:
            log(f"{cond:<25} {res['val_f1']:>8.4f} {res['test_f1']:>9.4f}")

    # ---- Detailed classification reports ------------------------------------
    log("\n--- Detailed Test Reports ---")
    for cond, res in results.items():
        if res is None:
            continue
        log(f"\n{cond}:")
        for report_line in res["report"].strip().split("\n"):
            log(f"  {report_line}")

    # ---- Direction interpretability -----------------------------------------
    log("\n--- Direction Interpretability ---")
    log("(Lower pos-neg gap = direction does not correlate with sentiment)")

    test_data = torch.load(os.path.join(CACHE_DIR, "z_tweet_test.pt"), map_location="cpu")
    z_test    = test_data["embeddings"]
    y_test    = test_data["labels"]

    direction_files = {
        "delta_star":   "delta_star.pt",
        "v_style":      "v_style.pt",
        "v_shift":      "v_shift.pt",
    }

    interp_results = {}
    for d_name, fname in direction_files.items():
        fpath = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        d = torch.load(fpath, map_location="cpu")
        interp_results[d_name] = direction_interpretability(z_test, y_test, d, d_name)

    # Also compute label-guided direction interpretability
    train_data = torch.load(os.path.join(CACHE_DIR, "z_tweet_train.pt"), map_location="cpu")
    formal_data = torch.load(os.path.join(CACHE_DIR, "z_formal.pt"), map_location="cpu")
    z_formal   = formal_data["embeddings"]

    try:
        from pipeline.clean import compute_label_guided_direction
        d_label = compute_label_guided_direction(
            train_data["embeddings"], train_data["labels"], z_formal
        )
        interp_results["label_guided"] = direction_interpretability(z_test, y_test, d_label, "label_guided")
    except Exception as e:
        log(f"  ⚠ label_guided direction not available: {e}")

    log("")
    log(print_interpretability_table(interp_results))

    # ---- Style axis linearity check -----------------------------------------
    log("\n--- Style Axis Linearity Check ---")
    log("(cosine(z + α*direction, formal_centroid) should be monotonic)")

    formal_centroid = F.normalize(z_formal.mean(0), dim=-1)
    sample_idx = torch.randperm(len(z_test))[:20]
    z_sample   = z_test[sample_idx]

    for d_name, fname in direction_files.items():
        fpath = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        d       = torch.load(fpath, map_location="cpu")
        lin_res = linearity_check(z_sample, d, formal_centroid)
        log(f"\n  {d_name}:")
        log(f"  {'alpha':>6} | {'cos(z+α·d, formal_centroid)':>28}")
        log(f"  {'-'*38}")
        for alpha, cos in lin_res.items():
            log(f"  {alpha:>6.2f} | {cos:>28.4f}")

    # ---- Zero-shot preservation check ---------------------------------------
    log("\n--- Zero-Shot Preservation Check ---")
    log("(mean cosine between cleaned and original FORMAL embeddings)")
    log("(Higher = less semantic content removed = better preservation)")

    for d_name, fname in direction_files.items():
        fpath = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        d   = torch.load(fpath, map_location="cpu")
        res = zero_shot_preservation_check(z_formal, d)
        log(f"  {d_name:<15} mean_cos={res['mean_cosine_preserved']:.4f} "
            f"min={res['min_cosine_preserved']:.4f} std={res['std_cosine_preserved']:.4f}")

    # ---- Key analysis -------------------------------------------------------
    log("\n--- Analysis ---")
    b1_f1  = results.get("B1 (raw)",          {}).get("test_f1", None) if results.get("B1 (raw)") else None
    b2_f1  = results.get("B2 (SAE)",          {}).get("test_f1", None) if results.get("B2 (SAE)") else None
    b25_f1 = results.get("B2.5 (mean-shift)", {}).get("test_f1", None) if results.get("B2.5 (mean-shift)") else None
    b3_f1  = results.get("B3 (SAE+CBDC)",     {}).get("test_f1", None) if results.get("B3 (SAE+CBDC)") else None
    c_f1   = results.get("C (label-guided)",  {}).get("test_f1", None) if results.get("C (label-guided)") else None

    if b3_f1 is not None and b2_f1 is not None:
        delta_cbdc = b3_f1 - b2_f1
        log(f"  CBDC contribution (B3 - B2): {delta_cbdc:+.4f}")
        if delta_cbdc > 0:
            log("  ✓ CBDC-PGD adds value over raw SAE projection")
        else:
            log("  ✗ CBDC adds no value — SAE direction alone is sufficient (or noise dominates)")

    if b2_f1 is not None and b25_f1 is not None:
        delta_sae_vs_shift = b2_f1 - b25_f1
        log(f"  SAE over mean-shift (B2 - B2.5): {delta_sae_vs_shift:+.4f}")
        if delta_sae_vs_shift > 0:
            log("  ✓ SAE finds a more accurate style direction than naive mean-shift")
        else:
            log("  ✗ SAE does not improve over mean-shift — style axis is simple")

    if b3_f1 is not None and b1_f1 is not None:
        overall_gain = b3_f1 - b1_f1
        log(f"  Overall gain vs baseline (B3 - B1): {overall_gain:+.4f}")
        if overall_gain > 0.005:
            log("  ✓ Style removal improves sentiment classification")
        elif overall_gain > -0.005:
            log("  ~ Neutral effect — style is not a major confound for FinBERT")
        else:
            log("  ✗ Style removal hurts — FinBERT uses style markers as sentiment cues")

    # ---- Save report --------------------------------------------------------
    report_path = os.path.join(RESULTS_DIR, "eval_report_new.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nFull report saved → {report_path}")


if __name__ == "__main__":
    main()
