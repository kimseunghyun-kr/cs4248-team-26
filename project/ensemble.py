"""
Ensemble: soft-vote over multiple fine-tuned transformer models.

Loads saved model checkpoints, runs inference, averages logits, picks argmax.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_roberta import LABEL_MAP, ID2LABEL, TweetDataset

logger = logging.getLogger("ensemble")


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
    parser = argparse.ArgumentParser(description="Ensemble transformer models")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--job_id", default="ensemble")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    # Model configs: pairs of (model_name, checkpoint_path, text_col)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model specs as model_name:checkpoint:text_col, "
                             "e.g. cardiffnlp/twitter-roberta-base-sentiment-latest:results/models/roberta_01/best_model.pt:text")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"ensemble_{args.job_id}.log")
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
    logger.info(f"Device: {device}")

    # --- Data ---
    train_df = pd.read_csv(args.train_csv, encoding="utf-8", encoding_errors="replace")
    test_df = pd.read_csv(args.test_csv, encoding="utf-8", encoding_errors="replace")
    _, val_df = train_test_split(
        train_df, test_size=args.val_size,
        stratify=train_df["sentiment"], random_state=args.random_state,
    )

    # --- Collect logits from each model ---
    val_logits_list = []
    test_logits_list = []
    val_labels = None
    test_labels = None

    for model_spec in args.models:
        parts = model_spec.split(":")
        model_name, ckpt_path, text_col = parts[0], parts[1], parts[2]
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

        # Individual model scores
        v_preds = v_logits.argmax(axis=1)
        t_preds = t_logits.argmax(axis=1)
        logger.info(f"  Individual val_f1={f1_score(val_labels, v_preds, average='macro'):.4f}"
                    f"  test_f1={f1_score(test_labels, t_preds, average='macro'):.4f}")

        del model
        torch.cuda.empty_cache()

    # --- Soft vote: average logits ---
    val_avg = np.mean(val_logits_list, axis=0)
    test_avg = np.mean(test_logits_list, axis=0)

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

    logger.info(f"\n{'='*70}")
    logger.info(f"ENSEMBLE RESULTS ({len(args.models)} models, soft vote)")
    logger.info(f"{'='*70}")
    logger.info(f"Val  : acc={val_acc:.4f}  f1_macro={val_f1:.4f}")
    logger.info(f"\n{val_report}")
    logger.info(f"Test : acc={test_acc:.4f}  f1_macro={test_f1:.4f}")
    logger.info(f"\n{test_report}")

    summary = {
        "job_id": args.job_id,
        "timestamp": datetime.now().isoformat(),
        "models": args.models,
        "val_accuracy": round(val_acc, 4),
        "val_f1_macro": round(val_f1, 4),
        "val_report": val_report,
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "test_report": test_report,
    }
    out_path = os.path.join(args.output_dir, f"ensemble_{args.job_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
