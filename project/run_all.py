"""
End-to-end runner for the sentiment-classification pipeline.

Pipeline:
  1. data/embed.py          — encode text, cache embeddings and tokenized splits
  2. pipeline/classify.py   — train a linear probe or fine-tune a transformer
  3. pipeline/evaluate.py   — write the matching evaluation report

Usage:
  python run_all.py
  python run_all.py --classifier linear
  python run_all.py --model finbert
  python run_all.py --only_phase 2
  python run_all.py --start_phase 2
"""

import argparse
import os
import subprocess
import sys
import time


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import MODEL_REGISTRY, get_model_name, model_slug


PHASES = [
    (1, "data/embed.py", "Embedding extraction"),
    (2, "pipeline/classify.py", "Classifier training + evaluation"),
    (3, "pipeline/evaluate.py", "Evaluation report"),
]


def run_phase(script_path: str, description: str, extra_env: dict, extra_args: list[str]) -> bool:
    abs_path = os.path.join(PROJECT_DIR, script_path)
    print(f"\n{'#' * 70}")
    print(f"# PHASE: {description}")
    print(f"# Script: {script_path}")
    print(f"{'#' * 70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-u", abs_path, *extra_args],
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONUNBUFFERED": "1", **extra_env},
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] Phase '{description}' failed (rc={result.returncode})")
        print("        Fix the issue and resume with --start_phase <N>.")
        return False

    print(f"\n[OK] Phase '{description}' completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the sentiment-classification pipeline.")
    parser.add_argument("--start_phase", type=int, default=1, help="Resume from this phase (1-3).")
    parser.add_argument("--only_phase", type=int, default=None, help="Run only this phase.")
    parser.add_argument("--classifier", choices=["linear", "transformer"], default="transformer")
    parser.add_argument(
        "--model",
        default="bert",
        help=f"Backbone encoder. Registry shortcuts: {list(MODEL_REGISTRY.keys())}. Or pass any HuggingFace model ID.",
    )
    parser.add_argument("--tokenizer", default=None, help="Optional custom tokenizer (HuggingFace ID).")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze_layers", type=int, default=4)
    parser.add_argument("--pooling", default="cls")
    parser.add_argument(
        "--input_mode",
        choices=["text", "text_plus_selected", "text_selected_pair"],
        default="text_plus_selected",
    )
    parser.add_argument("--head_type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--loss_name", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--train_embeddings", action="store_true")
    parser.add_argument("--use_time_of_tweet", action="store_true")
    parser.add_argument("--use_age_of_user", action="store_true")
    parser.add_argument("--use_country", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    args = parser.parse_args()

    hf_model_name = get_model_name(args.model)
    slug = model_slug(args.model)
    cache_dir = os.path.join(PROJECT_DIR, "cache", slug)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "results"), exist_ok=True)

    extra_env = {
        "MODEL_NAME": hf_model_name,
        "CACHE_DIR": cache_dir,
    }
    if args.tokenizer:
        extra_env["TOKENIZER_NAME"] = args.tokenizer

    phase_args = {
        1: [
            "--model_name", hf_model_name,
            "--batch_size", str(args.embed_batch_size),
            "--max_length", str(args.max_length),
        ],
        2: [
            "--classifier", args.classifier,
            "--model_name", hf_model_name,
            "--batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--max_length", str(args.max_length),
            "--n_epochs", str(args.epochs),
            "--encoder_lr", str(args.encoder_lr),
            "--classifier_lr", str(args.classifier_lr),
            "--weight_decay", str(args.weight_decay),
            "--warmup_ratio", str(args.warmup_ratio),
            "--dropout", str(args.dropout),
            "--unfreeze_layers", str(args.unfreeze_layers),
            "--pooling", str(args.pooling),
            "--input_mode", str(args.input_mode),
            "--head_type", str(args.head_type),
            "--hidden_dim", str(args.hidden_dim),
            "--loss_name", str(args.loss_name),
            "--focal_gamma", str(args.focal_gamma),
            "--patience", str(args.patience),
            "--label_smoothing", str(args.label_smoothing),
            "--grad_clip_norm", str(args.grad_clip_norm),
        ],
        3: [
            "--mode", args.classifier,
        ],
    }

    if args.train_embeddings:
        phase_args[2].append("--train_embeddings")
    if args.use_time_of_tweet:
        phase_args[2].append("--use_time_of_tweet")
    if args.use_age_of_user:
        phase_args[2].append("--use_age_of_user")
    if args.use_country:
        phase_args[2].append("--use_country")
    if args.no_class_weights:
        phase_args[2].append("--no_class_weights")

    if args.only_phase is not None:
        phases = [p for p in PHASES if p[0] == args.only_phase]
        if not phases:
            print(f"ERROR: Phase {args.only_phase} not found. Valid: 1-{len(PHASES)}")
            sys.exit(1)
    else:
        phases = [p for p in PHASES if p[0] >= args.start_phase]

    print("=" * 70)
    print("Sentiment Classification Pipeline")
    print("=" * 70)
    print(f"Classifier:    {args.classifier}")
    print(f"Model:         {hf_model_name}  (--model {args.model})")
    if args.tokenizer:
        print(f"Tokenizer:     {args.tokenizer}")
    print(f"Cache dir:     {cache_dir}")
    print(f"Max length:    {args.max_length}")
    print(f"Running phases: {[p[0] for p in phases]}")
    if args.classifier == "transformer":
        print(f"Epochs:        {args.epochs}")
        print(f"Unfreeze last: {args.unfreeze_layers} layer(s)")
        print(f"Input mode:    {args.input_mode}")
        print(f"Head:          {args.head_type}")
        print(f"Loss:          {args.loss_name}")

    total_start = time.time()
    for num, script, desc in phases:
        if not run_phase(script, f"[{num}/{len(PHASES)}] {desc}", extra_env, phase_args[num]):
            print(f"\nPipeline stopped at phase {num}.")
            print(f"To resume: python run_all.py --classifier {args.classifier} --start_phase {num}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    report_name = "transformer_eval_report.txt" if args.classifier == "transformer" else "linear_eval_report.txt"
    print(f"\n{'=' * 70}")
    print(f"All phases complete in {total_elapsed / 60:.1f} minutes.")
    print(f"Results: {os.path.join(PROJECT_DIR, 'results', report_name)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
