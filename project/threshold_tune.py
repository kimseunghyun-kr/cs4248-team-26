"""
Tune per-class decision thresholds on val set to maximize F1 macro.
Uses saved ensemble logits or single model logits.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(os.path.dirname(__file__)))
from train_roberta import LABEL_MAP, ID2LABEL, TweetDataset


def get_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.float().cpu().numpy())
            all_labels.extend(labels.tolist())
    return np.concatenate(all_logits, axis=0), np.array(all_labels)


def predict_with_bias(logits, bias):
    """Add per-class bias to logits before argmax."""
    adjusted = logits + np.array(bias)
    return adjusted.argmax(axis=1)


def grid_search_bias(logits, labels, steps=21):
    """Search for per-class bias that maximizes F1 macro on val."""
    # Search range: boost neutral from -1.0 to +1.0, adjust neg/pos
    best_f1 = 0
    best_bias = [0, 0, 0]

    # Focus search on neutral bias (class 1) since that's the weak class
    # Also search neg (class 0) and pos (class 2) adjustments
    neu_range = np.linspace(-0.5, 1.5, steps)
    neg_range = np.linspace(-0.5, 0.5, 11)
    pos_range = np.linspace(-0.5, 0.5, 11)

    for neu_b in neu_range:
        for neg_b in neg_range:
            for pos_b in pos_range:
                bias = [neg_b, neu_b, pos_b]
                preds = predict_with_bias(logits, bias)
                f1 = f1_score(labels, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_bias = bias

    return best_bias, best_f1


def main():
    parser = argparse.ArgumentParser(description="Threshold tuning for ensemble")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--job_id", default="threshold")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--models", nargs="+", required=True,
                        help="model_name:checkpoint:text_col")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"threshold_{args.job_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("threshold")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Data ---
    train_df = pd.read_csv(args.train_csv, encoding="utf-8", encoding_errors="replace")
    test_df = pd.read_csv(args.test_csv, encoding="utf-8", encoding_errors="replace")
    _, val_df = train_test_split(
        train_df, test_size=args.val_size,
        stratify=train_df["sentiment"], random_state=args.random_state,
    )

    # --- Collect logits ---
    val_logits_list = []
    test_logits_list = []
    val_labels = None
    test_labels = None

    for model_spec in args.models:
        model_name, ckpt_path, text_col = model_spec.split(":")
        logger.info(f"Loading: {model_name} | text_col={text_col}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, ignore_mismatched_sizes=True,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device)

        val_texts = val_df[text_col].fillna("").tolist()
        test_texts = test_df[text_col].fillna("").tolist()
        v_labels = [LABEL_MAP[s] for s in val_df["sentiment"]]
        t_labels = [LABEL_MAP[s] for s in test_df["sentiment"]]

        val_ds = TweetDataset(val_texts, v_labels, tokenizer, args.max_length)
        test_ds = TweetDataset(test_texts, t_labels, tokenizer, args.max_length)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        v_logits, v_lab = get_logits(model, val_loader, device)
        t_logits, t_lab = get_logits(model, test_loader, device)

        val_logits_list.append(v_logits)
        test_logits_list.append(t_logits)
        val_labels = v_lab
        test_labels = t_lab

        del model
        torch.cuda.empty_cache()

    # --- Average logits ---
    val_avg = np.mean(val_logits_list, axis=0)
    test_avg = np.mean(test_logits_list, axis=0)

    # --- Baseline (no bias) ---
    base_val_preds = val_avg.argmax(axis=1)
    base_test_preds = test_avg.argmax(axis=1)
    base_val_f1 = f1_score(val_labels, base_val_preds, average="macro")
    base_test_f1 = f1_score(test_labels, base_test_preds, average="macro")
    logger.info(f"\nBaseline (no bias): val_f1={base_val_f1:.4f}  test_f1={base_test_f1:.4f}")
    logger.info(classification_report(test_labels, base_test_preds,
                                      target_names=["negative", "neutral", "positive"]))

    # --- Grid search on val ---
    logger.info("Searching for optimal per-class bias on val set...")
    best_bias, best_val_f1 = grid_search_bias(val_avg, val_labels)
    logger.info(f"Best bias: neg={best_bias[0]:.2f}  neu={best_bias[1]:.2f}  pos={best_bias[2]:.2f}")
    logger.info(f"Best val_f1: {best_val_f1:.4f} (was {base_val_f1:.4f})")

    # --- Apply to test ---
    tuned_val_preds = predict_with_bias(val_avg, best_bias)
    tuned_test_preds = predict_with_bias(test_avg, best_bias)

    tuned_test_f1 = f1_score(test_labels, tuned_test_preds, average="macro")
    tuned_test_acc = accuracy_score(test_labels, tuned_test_preds)
    tuned_val_f1 = f1_score(val_labels, tuned_val_preds, average="macro")

    val_report = classification_report(val_labels, tuned_val_preds,
                                       target_names=["negative", "neutral", "positive"])
    test_report = classification_report(test_labels, tuned_test_preds,
                                        target_names=["negative", "neutral", "positive"])

    logger.info(f"\n{'='*70}")
    logger.info("THRESHOLD-TUNED RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Bias: neg={best_bias[0]:.2f}  neu={best_bias[1]:.2f}  pos={best_bias[2]:.2f}")
    logger.info(f"Val  : f1_macro={tuned_val_f1:.4f}")
    logger.info(f"\n{val_report}")
    logger.info(f"Test : acc={tuned_test_acc:.4f}  f1_macro={tuned_test_f1:.4f}")
    logger.info(f"\n{test_report}")
    logger.info(f"Gain over baseline: {tuned_test_f1 - base_test_f1:+.4f}")

    summary = {
        "job_id": args.job_id,
        "timestamp": datetime.now().isoformat(),
        "models": args.models,
        "bias": {"negative": best_bias[0], "neutral": best_bias[1], "positive": best_bias[2]},
        "baseline_val_f1": round(base_val_f1, 4),
        "baseline_test_f1": round(base_test_f1, 4),
        "tuned_val_f1": round(tuned_val_f1, 4),
        "tuned_test_f1": round(tuned_test_f1, 4),
        "tuned_test_acc": round(tuned_test_acc, 4),
        "val_report": val_report,
        "test_report": test_report,
    }
    out_path = os.path.join(args.output_dir, f"threshold_{args.job_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
