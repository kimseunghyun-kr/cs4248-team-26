"""
Phase 5: Full evaluation report — direction interpretability + analysis.

Loads results from classify.py and produces:
  1. Summary table (all conditions)
  2. Direction interpretability per sentiment class
  3. Confound axis linearity check
  4. Zero-shot preservation on formal corpus
  5. Comparative analysis (D1 vs B1, D2 vs B1, D3 vs B1)

Run from project/ directory:
  python pipeline/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from pipeline.clean import project_out

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR   = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def direction_interpretability(z, labels, direction, direction_name):
    """Mean projection score per sentiment class onto a direction."""
    if direction.dim() == 2:
        direction = direction[0]  # use first principal component
    d = F.normalize(direction, dim=-1)
    proj = (z @ d).numpy()
    labels_np = labels.numpy()

    result = {}
    for c, name in enumerate(["negative", "neutral", "positive"]):
        mask = labels_np == c
        result[name] = float(proj[mask].mean()) if mask.sum() > 0 else float("nan")
    result["pos_neg_gap"] = abs(result["positive"] - result["negative"])
    return result


def linearity_check(z_sample, direction, formal_centroid):
    """cosine(z + alpha*d, formal_centroid) vs alpha — should be monotonic."""
    if direction.dim() == 2:
        direction = direction[0]
    d = F.normalize(direction, dim=-1)
    fc = F.normalize(formal_centroid, dim=-1)
    alphas = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    results = {}
    for a in alphas:
        z_s = F.normalize(z_sample + a * d.unsqueeze(0), dim=-1)
        results[a] = F.cosine_similarity(z_s, fc.unsqueeze(0), dim=-1).mean().item()
    return results


def zero_shot_preservation(z_formal, direction):
    """Cosine between cleaned and original formal embeddings."""
    z_clean = project_out(z_formal, direction)
    cos = F.cosine_similarity(z_clean, z_formal, dim=-1)
    return {
        "mean": cos.mean().item(),
        "min":  cos.min().item(),
        "std":  cos.std().item(),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("CBDC Financial Sentiment — Full Evaluation Report")
    log("=" * 70)

    # ---- Classification results -----------------------------------------------
    results_path = os.path.join(CACHE_DIR, "results.pt")
    if not os.path.exists(results_path):
        log("ERROR: results.pt not found. Run classify.py first.")
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

    # ---- Detailed reports -----------------------------------------------------
    log("\n--- Detailed Test Reports ---")
    for cond, res in results.items():
        if res is None:
            continue
        log(f"\n{cond}:")
        for line in res["report"].strip().split("\n"):
            log(f"  {line}")

    # ---- Direction interpretability -------------------------------------------
    log("\n--- Direction Interpretability ---")
    log("(Lower pos-neg gap = direction is sentiment-neutral = better confound direction)")

    test_path = os.path.join(CACHE_DIR, "z_tweet_test.pt")
    if not os.path.exists(test_path):
        log("  [skip] z_tweet_test.pt not found")
    else:
        test_data = torch.load(test_path, map_location="cpu")
        z_test = test_data["embeddings"]
        y_test = test_data["labels"]

        direction_files = {"cbdc_directions": "cbdc_directions.pt"}

        interp_results = {}
        for d_name, fname in direction_files.items():
            fpath = os.path.join(CACHE_DIR, fname)
            if not os.path.exists(fpath):
                continue
            d = torch.load(fpath, map_location="cpu")
            interp_results[d_name] = direction_interpretability(z_test, y_test, d, d_name)

        header = f"{'Direction':<20} {'neg_mean':>10} {'neu_mean':>10} {'pos_mean':>10} {'gap':>8}"
        log(f"\n{header}")
        log("-" * len(header))
        for d_name, row in interp_results.items():
            log(f"{d_name:<20} {row['negative']:>10.4f} {row['neutral']:>10.4f} "
                f"{row['positive']:>10.4f} {row['pos_neg_gap']:>8.4f}")

    # ---- Linearity check ------------------------------------------------------
    formal_path = os.path.join(CACHE_DIR, "z_formal.pt")
    if os.path.exists(test_path) and os.path.exists(formal_path):
        z_formal = torch.load(formal_path, map_location="cpu")["embeddings"]
        formal_centroid = F.normalize(z_formal.mean(0), dim=-1)
        sample_idx = torch.randperm(len(z_test))[:20]
        z_sample = z_test[sample_idx]

        log("\n--- Confound Axis Linearity Check ---")
        log("(cosine(z + alpha*d, formal_centroid) — monotonic = good axis)")

        for d_name, fname in direction_files.items():
            fpath = os.path.join(CACHE_DIR, fname)
            if not os.path.exists(fpath):
                continue
            d = torch.load(fpath, map_location="cpu")
            lin = linearity_check(z_sample, d, formal_centroid)
            log(f"\n  {d_name}:")
            log(f"  {'alpha':>6} | {'cos':>8}")
            log(f"  {'-'*18}")
            for a, c in lin.items():
                log(f"  {a:>6.2f} | {c:>8.4f}")

        # ---- Zero-shot preservation -------------------------------------------
        log("\n--- Zero-Shot Preservation Check ---")
        log("(cosine between cleaned and original FORMAL embeddings)")

        for d_name, fname in direction_files.items():
            fpath = os.path.join(CACHE_DIR, fname)
            if not os.path.exists(fpath):
                continue
            d = torch.load(fpath, map_location="cpu")
            zs = zero_shot_preservation(z_formal, d)
            log(f"  {d_name:<20} mean={zs['mean']:.4f} min={zs['min']:.4f} std={zs['std']:.4f}")

    # ---- Comparative analysis -------------------------------------------------
    log("\n--- Comparative Analysis ---")

    def _f1(key):
        r = results.get(key)
        return r["test_f1"] if r else None

    b1 = _f1("B1 (raw)")
    d1 = _f1("D1 (debias_vl)")
    d2 = _f1("D2 (CBDC)")
    d3 = _f1("D3 (CBDC+proj)")
    c  = _f1("C (label-guided)")

    if b1 is not None and d1 is not None:
        delta = d1 - b1
        log(f"  debias_vl vs baseline (D1 - B1): {delta:+.4f}")
        if delta > 0.005:
            log("    -> Word-pair projection improves classification")
        elif delta > -0.005:
            log("    -> Neutral effect")
        else:
            log("    -> Word-pair projection hurts — prompts may not capture real confounds")

    if b1 is not None and d2 is not None:
        delta = d2 - b1
        log(f"  CBDC encoder vs baseline (D2 - B1): {delta:+.4f}")
        if delta > 0.005:
            log("    -> CBDC text_iccv training improves classification")
        elif delta > -0.005:
            log("    -> Neutral effect")
        else:
            log("    -> CBDC training hurts — may need different prompts or model")

    if b1 is not None and d3 is not None:
        delta = d3 - b1
        log(f"  CBDC+projection vs baseline (D3 - B1): {delta:+.4f}")

    if d2 is not None and d3 is not None:
        delta = d3 - d2
        log(f"  Residual projection benefit (D3 - D2): {delta:+.4f}")
        if delta > 0:
            log("    -> Residual projection removes additional confound from CBDC encoder")
        else:
            log("    -> Encoder already learned orthogonality")

    if d1 is not None and d2 is not None:
        delta = d2 - d1
        log(f"  CBDC vs debias_vl (D2 - D1): {delta:+.4f}")
        if delta > 0:
            log("    -> PGD refinement + training outperforms closed-form projection")
        else:
            log("    -> Closed-form debias_vl is sufficient")

    if b1 is not None and c is not None:
        delta = c - b1
        log(f"  Oracle (label-guided) vs baseline (C - B1): {delta:+.4f}")

    # ---- Save -----------------------------------------------------------------
    report_path = os.path.join(RESULTS_DIR, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nFull report saved -> {report_path}")


if __name__ == "__main__":
    main()
