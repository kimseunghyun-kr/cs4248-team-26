"""
Weighted ensemble: grid-search over model weights on val set, evaluate on test.
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

from train_roberta import LABEL_MAP, ID2LABEL, TweetDataset

logger = logging.getLogger("weighted_ensemble")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--job_id", default="weighted_ens")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--models", nargs="+", required=True,
                        help="model_name:checkpoint:text_col")
    parser.add_argument("--weight_step", type=float, default=0.1,
                        help="Step size for weight grid search (default 0.1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"weighted_ens_{args.job_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f"Job ID: {args.job_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model_names = []

    for model_spec in args.models:
        parts = model_spec.split(":")
        model_name, ckpt_path, text_col = parts[0], parts[1], parts[2]
        model_names.append(model_name.split("/")[-1])
        logger.info(f"\nLoading: {model_name} | ckpt={ckpt_path} | text_col={text_col}")

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

        v_logits, v_labels = get_logits(model, val_loader, device)
        t_logits, t_labels = get_logits(model, test_loader, device)

        val_logits_list.append(v_logits)
        test_logits_list.append(t_logits)
        val_labels = v_labels
        test_labels = t_labels

        v_preds = v_logits.argmax(axis=1)
        t_preds = t_logits.argmax(axis=1)
        logger.info(f"  Individual val_f1={f1_score(val_labels, v_preds, average='macro'):.4f}"
                    f"  test_f1={f1_score(test_labels, t_preds, average='macro'):.4f}")

        del model
        torch.cuda.empty_cache()

    # --- Grid search weights on val set ---
    n_models = len(args.models)
    step = args.weight_step
    # Generate weight candidates that sum to 1
    steps = int(round(1.0 / step))
    weight_values = [round(i * step, 2) for i in range(1, steps + 1)]  # exclude 0

    logger.info(f"\nSearching weights with step={step} over {n_models} models...")

    best_val_f1 = 0
    best_weights = None
    count = 0

    # Generate all combinations where weights sum to 1
    for combo in product(weight_values, repeat=n_models):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        count += 1
        weights = np.array(combo)
        avg_logits = sum(w * l for w, l in zip(weights, val_logits_list))
        preds = avg_logits.argmax(axis=1)
        f1 = f1_score(val_labels, preds, average="macro")
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_weights = combo

    logger.info(f"Searched {count} weight combinations")
    logger.info(f"Best val weights: {best_weights} -> val_f1={best_val_f1:.4f}")
    for name, w in zip(model_names, best_weights):
        logger.info(f"  {name}: {w:.2f}")

    # --- Evaluate best weights on test ---
    weights = np.array(best_weights)
    test_avg = sum(w * l for w, l in zip(weights, test_logits_list))
    val_avg = sum(w * l for w, l in zip(weights, val_logits_list))

    val_preds = val_avg.argmax(axis=1)
    test_preds = test_avg.argmax(axis=1)

    val_f1 = f1_score(val_labels, val_preds, average="macro")
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    val_acc = accuracy_score(val_labels, val_preds)
    test_acc = accuracy_score(test_labels, test_preds)

    val_report = classification_report(val_labels, val_preds,
                                       target_names=["negative", "neutral", "positive"])
    test_report = classification_report(test_labels, test_preds,
                                        target_names=["negative", "neutral", "positive"])

    # --- Also show uniform for comparison ---
    uni_test_avg = np.mean(test_logits_list, axis=0)
    uni_test_preds = uni_test_avg.argmax(axis=1)
    uni_test_f1 = f1_score(test_labels, uni_test_preds, average="macro")

    logger.info(f"\n{'='*70}")
    logger.info(f"WEIGHTED ENSEMBLE RESULTS ({n_models} models)")
    logger.info(f"{'='*70}")
    logger.info(f"Weights: {best_weights}")
    logger.info(f"Val  : acc={val_acc:.4f}  f1_macro={val_f1:.4f}")
    logger.info(f"\n{val_report}")
    logger.info(f"Test : acc={test_acc:.4f}  f1_macro={test_f1:.4f}")
    logger.info(f"\n{test_report}")
    logger.info(f"Uniform ensemble test_f1={uni_test_f1:.4f} (for comparison)")

    summary = {
        "job_id": args.job_id,
        "timestamp": datetime.now().isoformat(),
        "models": args.models,
        "weights": list(best_weights),
        "model_names": model_names,
        "val_accuracy": round(val_acc, 4),
        "val_f1_macro": round(val_f1, 4),
        "val_report": val_report,
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "test_report": test_report,
        "uniform_test_f1": round(uni_test_f1, 4),
    }
    out_path = os.path.join(args.output_dir, f"weighted_ens_{args.job_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
