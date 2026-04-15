"""
Fine-tune cardiffnlp/twitter-roberta-base-sentiment-latest for tweet sentiment.

Usage:
    python project/train_roberta.py \
        --train_csv data/train_cleaned.csv \
        --test_csv  data/test_cleaned.csv \
        --job_id    roberta_01
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train_epoch(model, loader, optimizer, scheduler, device, use_amp=True,
                criterion=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels) if criterion else \
                    torch.nn.functional.cross_entropy(outputs.logits, labels)
            loss.backward()
        else:
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels) if criterion else \
                torch.nn.functional.cross_entropy(outputs.logits, labels)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        total_loss += outputs.loss.item() * len(labels)
        total += len(labels)
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    return {
        "loss": total_loss / total,
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "report": classification_report(
            labels, preds,
            target_names=["negative", "neutral", "positive"],
            zero_division=0,
        ),
        "preds": preds,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Twitter-RoBERTa")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--text_col", default="cleaned_text")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--job_id", default="roberta")
    parser.add_argument("--output_dir", default="results")
    # Model
    parser.add_argument("--model_name", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--max_length", type=int, default=128)
    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision (needed for DeBERTa-v3)")
    parser.add_argument("--class_weighted", action="store_true",
                        help="Use balanced class weights in loss")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"roberta_{args.job_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("train_roberta")
    logger.info(f"Job ID : {args.job_id}")
    logger.info(f"Args   : {json.dumps(vars(args), indent=2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    if device.type == "cuda":
        logger.info(f"GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Data ---
    train_df = pd.read_csv(args.train_csv, encoding="utf-8", encoding_errors="replace")
    test_df = pd.read_csv(args.test_csv, encoding="utf-8", encoding_errors="replace")

    train_df, val_df = train_test_split(
        train_df, test_size=args.val_size,
        stratify=train_df["sentiment"], random_state=args.random_state,
    )

    train_texts = train_df[args.text_col].fillna("").tolist()
    val_texts = val_df[args.text_col].fillna("").tolist()
    test_texts = test_df[args.text_col].fillna("").tolist()

    train_labels = [LABEL_MAP[s] for s in train_df["sentiment"]]
    val_labels = [LABEL_MAP[s] for s in val_df["sentiment"]]
    test_labels = [LABEL_MAP[s] for s in test_df["sentiment"]]

    logger.info(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # --- Tokenizer & Model ---
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
        ignore_mismatched_sizes=True,
    )
    # DeBERTa-v3 fix: word embeddings are stored in FP16 which causes NaN
    # during mixed-precision training. Cast them to FP32 explicitly.
    if "deberta" in args.model_name.lower():
        logger.info("Applying DeBERTa-v3 FP16 embedding fix...")
        # Force all params to FP32 first
        model.float()
        # Specifically ensure embeddings are FP32 (the root cause of NaN)
        for name, param in model.named_parameters():
            if "embedding" in name.lower():
                param.data = param.data.float()
        # Disable AMP for DeBERTa — bfloat16 autocast still triggers NaN
        # in the disentangled attention computation
        if not args.no_amp:
            logger.info("  Forcing --no_amp for DeBERTa stability")
            args.no_amp = True
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {param_count:,}")

    # --- Datasets & Loaders ---
    logger.info("Tokenizing...")
    train_ds = TweetDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_ds = TweetDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_ds = TweetDataset(test_texts, test_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=0, pin_memory=True)

    # --- Optimizer & Scheduler ---
    # Use higher LR for classifier head (randomly initialized) vs backbone
    no_decay = ["bias", "LayerNorm.weight"]
    classifier_names = ["classifier", "pooler"]
    optimizer_grouped = [
        # Backbone params with decay
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(cn in n for cn in classifier_names)],
         "weight_decay": args.weight_decay, "lr": args.lr},
        # Backbone params without decay
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(cn in n for cn in classifier_names)],
         "weight_decay": 0.0, "lr": args.lr},
        # Classifier head (higher LR)
        {"params": [p for n, p in model.named_parameters()
                    if any(cn in n for cn in classifier_names)],
         "weight_decay": args.weight_decay, "lr": args.lr * 10},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"Total steps: {total_steps} | Warmup: {warmup_steps}")

    # --- Class-weighted loss ---
    criterion = None
    if args.class_weighted:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(train_labels)
        cw = compute_class_weight("balanced", classes=classes, y=np.array(train_labels))
        cw_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=cw_tensor)
        logger.info(f"Class weights: {dict(zip(['neg','neu','pos'], cw.round(3)))}")

    # --- Training ---
    best_val_f1 = 0.0
    best_state = None
    model_save_dir = os.path.join(args.output_dir, "models", args.job_id)
    os.makedirs(model_save_dir, exist_ok=True)

    wall_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        use_amp = not args.no_amp
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device,
                                            use_amp=use_amp, criterion=criterion)
        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{args.epochs} | {elapsed:.0f}s | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> New best val_f1={best_val_f1:.4f}")

    # --- Load best & evaluate on test ---
    model.load_state_dict(best_state)
    model.to(device)

    use_amp = not args.no_amp
    val_final = evaluate(model, val_loader, device, use_amp=use_amp)
    test_final = evaluate(model, test_loader, device, use_amp=use_amp)
    total_elapsed = time.time() - wall_start

    logger.info(f"\n{'='*70}")
    logger.info("FINAL RESULTS (best checkpoint)")
    logger.info(f"{'='*70}")
    logger.info(f"Val  : acc={val_final['accuracy']:.4f} f1_macro={val_final['f1_macro']:.4f}")
    logger.info(f"\n{val_final['report']}")
    logger.info(f"Test : acc={test_final['accuracy']:.4f} f1_macro={test_final['f1_macro']:.4f}")
    logger.info(f"\n{test_final['report']}")
    logger.info(f"Total time: {total_elapsed:.0f}s")

    # --- Save model ---
    save_path = os.path.join(model_save_dir, "best_model.pt")
    torch.save(best_state, save_path)
    logger.info(f"Model saved -> {save_path}")

    # --- Save summary JSON ---
    summary = {
        "job_id": args.job_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "device": str(device),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "epochs": args.epochs,
        "best_epoch_val_f1": round(best_val_f1, 4),
        "val_accuracy": round(val_final["accuracy"], 4),
        "val_f1_macro": round(val_final["f1_macro"], 4),
        "val_f1_weighted": round(val_final["f1_weighted"], 4),
        "val_report": val_final["report"],
        "test_accuracy": round(test_final["accuracy"], 4),
        "test_f1_macro": round(test_final["f1_macro"], 4),
        "test_f1_weighted": round(test_final["f1_weighted"], 4),
        "test_report": test_final["report"],
        "args": vars(args),
    }

    out_path = os.path.join(args.output_dir, f"roberta_{args.job_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved -> {out_path}")


if __name__ == "__main__":
    main()
