"""
Training loop for CBDCSentimentClassifier.

Only the MLP head is trained — FinBERT backbone is fully frozen.
Saves best checkpoint by val macro F1.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from typing import Optional

from model import CBDCSentimentClassifier
from config import TrainConfig


# ---------------------------------------------------------------------------
# Scheduler: cosine with linear warmup
# ---------------------------------------------------------------------------
def get_cosine_schedule_with_warmup(optimizer, n_warmup_steps, n_total_steps):
    def lr_lambda(step):
        if step < n_warmup_steps:
            return float(step) / max(1, n_warmup_steps)
        progress = float(step - n_warmup_steps) / max(1, n_total_steps - n_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------
def train_epoch(
    model: CBDCSentimentClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    return total_loss / n


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: CBDCSentimentClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "acc": acc, "f1_macro": f1}


# ---------------------------------------------------------------------------
# Full training procedure
# ---------------------------------------------------------------------------
def train(
    model: CBDCSentimentClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    label: str = "full",          # label for checkpoint filename
) -> dict:
    """
    Returns dict with training history and path to best checkpoint.
    """
    device = cfg.device

    # Only optimize MLP head parameters
    trainable = list(model.head.parameters())
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_total_steps = len(train_loader) * cfg.epochs
    n_warmup = int(n_total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, n_total_steps)

    criterion = nn.CrossEntropyLoss()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_f1 = -1.0
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, f"best_model_{label}.pt")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

        print(
            f"  epoch {epoch:>2}/{cfg.epochs} | "
            f"loss {train_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | "
            f"val_acc {val_metrics['acc']:.4f} | "
            f"val_f1 {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "epoch": epoch,
                    "head_state_dict": model.head.state_dict(),
                    "val_f1_macro": best_f1,
                    "ablation": model.ablation,
                },
                best_ckpt_path,
            )

    print(f"  Best val F1: {best_f1:.4f}  checkpoint → {best_ckpt_path}")
    return {"history": history, "best_f1": best_f1, "ckpt": best_ckpt_path}
