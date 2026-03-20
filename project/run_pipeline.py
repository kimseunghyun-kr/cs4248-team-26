"""
End-to-end pipeline:
  1. Load dataset
  2. Build direction bank from train split (or load if cached)
  3. Train full model + ablation variants
  4. Evaluate on test set
  5. Print final report
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve device early so configs can reference it
# ---------------------------------------------------------------------------
def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _get_device()
print(f"Using device: {DEVICE}")

from config import PGDConfig, TrainConfig
from encoder import FinBERTEncoder
from direction_bank import FinancialDirectionBank
from dataset import load_tsad, TweetDataset, make_collate_fn
from model import CBDCSentimentClassifier
from train import train
from evaluate import (
    run_test_evaluation,
    direction_interpretability,
    print_classification_report,
    print_confusion_matrix,
    print_interpretability_table,
    print_ablation,
    save_report,
    load_model,
)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Config ---
    cfg = TrainConfig()
    cfg.device = DEVICE

    pgd_cfg = PGDConfig()
    pgd_cfg.device = DEVICE

    set_seed(cfg.seed)

    # --- Encoder ---
    try:
        encoder = FinBERTEncoder(model_name=cfg.model_name, device=DEVICE)
    except Exception as e:
        print(f"Could not load {cfg.model_name}: {e}")
        print(f"Falling back to {cfg.fallback_model_name}")
        encoder = FinBERTEncoder(model_name=cfg.fallback_model_name, device=DEVICE)
        cfg.model_name = cfg.fallback_model_name

    cfg.hidden_size = encoder.hidden_size

    # --- Dataset ---
    (
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
    ) = load_tsad(cfg.dataset_name, cfg.dataset_config)

    tokenizer = encoder.tokenizer
    pad_id = tokenizer.pad_token_id or 0

    train_ds = TweetDataset(train_texts, train_labels, tokenizer)
    val_ds   = TweetDataset(val_texts,   val_labels,   tokenizer)
    test_ds  = TweetDataset(test_texts,  test_labels,  tokenizer)

    collate = make_collate_fn(pad_id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0)

    # --- Direction Bank ---
    bank_path = cfg.direction_bank_path

    if os.path.exists(bank_path):
        print(f"\nFound cached direction bank at '{bank_path}'. Loading ...")
        bank = FinancialDirectionBank.load(bank_path)
    else:
        print(f"\nBuilding direction bank from first {pgd_cfg.n_anchors} train texts ...")
        anchor_texts = train_texts[: pgd_cfg.n_anchors]
        bank = FinancialDirectionBank(hidden_size=cfg.hidden_size)
        bank.build(encoder, anchor_texts, pgd_cfg)
        bank.save(bank_path)

    bank.to(DEVICE)
    cfg.n_concept_axes = len(bank)
    cfg.n_directions_per_axis = pgd_cfg.n_directions

    print(f"\nDirection Bank: {len(bank)} axes × {pgd_cfg.n_directions} directions each")
    for name in bank.axis_names:
        print(f"  {name}")

    # --- Train ablation variants ---
    ablation_results = {}

    for ablation in ["full", "z_only", "directions_only"]:
        label_display = {
            "full": "z + directions (full model)",
            "z_only": "z only",
            "directions_only": "directions only",
        }[ablation]

        print(f"\n{'='*60}")
        print(f"Training: {label_display}")
        print(f"{'='*60}")

        model = CBDCSentimentClassifier(encoder, bank, cfg, ablation=ablation)
        model.to(DEVICE)

        result = train(model, train_loader, val_loader, cfg, label=ablation)
        ablation_results[ablation] = result["best_f1"]

    # --- Evaluate best full model on test set ---
    print(f"\n{'='*60}")
    print("Test Evaluation (full model)")
    print(f"{'='*60}")

    full_ckpt = os.path.join(cfg.checkpoint_dir, "best_model_full.pt")
    full_model = load_model(full_ckpt, encoder, bank, cfg, ablation="full")
    full_model.to(DEVICE)

    test_results = run_test_evaluation(full_model, test_loader, DEVICE)
    interp_table = direction_interpretability(full_model, test_loader, DEVICE)

    # --- Print results ---
    print("\nTest Results:")
    print_classification_report(test_results["report"])
    print_confusion_matrix(test_results["confusion_matrix"])

    print("\nAblation:")
    print_ablation(ablation_results)

    print("\nDirection Interpretability:")
    print_interpretability_table(interp_table)

    # --- Save report ---
    save_report(cfg.results_dir, test_results, interp_table, ablation_results)


if __name__ == "__main__":
    main()
