"""
Phase 4: Evaluate TRUE ZERO-SHOT for all experiment conditions.

Conditions:
  B1 (raw)          : raw BERT-derivative CLS embeddings (baseline)
  D1 (debias_vl)    : debias_vl word-pair projection applied to raw embeddings
  D2 (CBDC)         : CBDC text_iccv fine-tuned encoder embeddings
  D3 (CBDC+proj)    : CBDC encoder + residual CBDC direction projection
  D4 (raw+sent-boost)  : raw embeddings + CBDC confound removal + sentiment boost
  D5 (CBDC+sent-boost) : CBDC embeddings + sentiment boost
  C (label-guided)  : label-guided within-class mean-shift projection (oracle)

All conditions evaluate using zero-shot Cosine Similarity against sentiment prototypes.

Run from project/ directory:
  python pipeline/classify.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv # <--- Add this at the top of classify.py
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
import numpy as np

from pipeline.clean import materialize_sentiment_boost_conditions

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


def evaluate_zero_shot(z_tweets, y_true, texts, z_prototypes, device):
    """
    Perform true zero-shot classification and track disagreements.
    """
    z_tweets = z_tweets.to(device)
    z_prototypes = z_prototypes.to(device)

    # 1. Calculate Cosine Similarity
    z_norm = F.normalize(z_tweets, dim=-1)
    proto_norm = F.normalize(z_prototypes, dim=-1)
    sim = z_norm @ proto_norm.T  

    # 2. Get predictions
    preds = sim.argmax(dim=-1).cpu().numpy()
    labels_np = y_true.numpy()

    # 3. Calculate metrics
    f1 = f1_score(labels_np, preds, average="macro")
    report = classification_report(
        labels_np, preds,
        target_names=["negative", "neutral", "positive"],
        digits=4,
        zero_division=0
    )

    # 4. Catch the disagreements
    disagreements = []
    if texts is not None:
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        for i in range(len(preds)):
            if preds[i] != labels_np[i]:
                disagreements.append({
                    "text": texts[i],
                    "true_label": label_map.get(labels_np[i], "unknown"),
                    "pred_label": label_map.get(preds[i], "unknown")
                })

    return f1, report, disagreements


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
    "D4 (raw+sent-boost)": "clean_raw_sentiment_boost",
    "D5 (CBDC+sent-boost)": "clean_cbdc_sentiment_boost",
    "C (label-guided)":    "clean_label_guided",
}

BOOST_SUFFIXES = {
    "clean_raw_sentiment_boost",
    "clean_cbdc_sentiment_boost",
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # LOAD ZERO-SHOT PROTOTYPES
    # ---------------------------------------------------------
    proto_path = os.path.join(CACHE_DIR, "sentiment_prototypes.pt")
    if not os.path.exists(proto_path):
        print(f"CRITICAL ERROR: Cannot find sentiment prototypes at {proto_path}")
        print("Please ensure Phase 2 (refine.py) has run successfully.")
        return

    # Shape should be (3, H) for Negative, Neutral, Positive
    z_prototypes = torch.load(proto_path, map_location="cpu")
    print(f"\nLoaded Zero-Shot Sentiment Prototypes: Shape {tuple(z_prototypes.shape)}")

    all_results = {}

    for cond_name, cache_suffix in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (Zero-Shot)")
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

        # In Zero-Shot, we do not train on the train set. We just evaluate.
        z_val   = data["val"]["embeddings"]
        y_val   = data["val"]["labels"]
        z_test  = data["test"]["embeddings"]
        y_test  = data["test"]["labels"]
        
        # --- NEW: Extract texts if they exist in the cache ---
        texts_test = data["test"].get("texts", None)

        print(f"  val={len(z_val)} test={len(z_test)} dim={z_val.shape[1]}")

        # --- Evaluate Zero-Shot Cosine Similarity ---
        # Pass 'None' for validation texts, and 'texts_test' for the test set
        val_f1, _, _ = evaluate_zero_shot(z_val, y_val, None, z_prototypes, device)
        test_f1, report, disagreements = evaluate_zero_shot(z_test, y_test, texts_test, z_prototypes, device)

        print(f"  val_f1={val_f1:.4f} | test_f1={test_f1:.4f}")
        
        # --- NEW: Save disagreements to CSV for manual review ---
        import csv
        if disagreements:
            csv_path = os.path.join(CACHE_DIR, f"disagreements_{cond_name[:2]}.csv")
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["text", "true_label", "pred_label"])
                writer.writeheader()
                writer.writerows(disagreements)
            print(f"  Saved {len(disagreements)} disagreeing tweets to -> {csv_path}")

        print(f"\n  Test classification report:")
        for line in report.strip().split("\n"):
            print(f"    {line}")

        all_results[cond_name] = {
            "val_f1":  val_f1,
            "test_f1": test_f1,
            "report":  report,
        }

    # Summary
    print(f"\n{'='*60}")
    print("ZERO-SHOT SUMMARY")
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