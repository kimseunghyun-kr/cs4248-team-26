"""
Separate entry pipe for transformer fine-tuning.

Reuses the existing project phases:
  1. data/embed.py          — cache token IDs / masks / embeddings
  2. pipeline/classify.py   — transformer fine-tuning mode
  3. pipeline/evaluate.py   — transformer evaluation mode
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
    (2, "pipeline/classify.py", "Transformer classifier training + eval"),
    (3, "pipeline/evaluate.py", "Transformer evaluation report"),
]


def run_phase(script_path: str, description: str, extra_env: dict, extra_args: list[str]) -> bool:
    abs_path = os.path.join(PROJECT_DIR, script_path)
    print(f"\n{'#'*70}")
    print(f"# PHASE: {description}")
    print(f"# Script: {script_path}")
    print(f"{'#'*70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-u", abs_path, *extra_args],
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONUNBUFFERED": "1", **extra_env},
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] Phase '{description}' failed (rc={result.returncode})")
        return False

    print(f"\n[OK] Phase '{description}' completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run transformer fine-tuning pipeline.")
    parser.add_argument("--start_phase", type=int, default=1, help="Resume from this phase (1-3).")
    parser.add_argument("--only_phase", type=int, default=None, help="Run only this phase.")
    parser.add_argument("--model", default="bert",
                        help=f"Backbone encoder. Registry shortcuts: {list(MODEL_REGISTRY.keys())}. "
                             "Or pass any HuggingFace model ID.")
    parser.add_argument("--tokenizer", default=None,
                        help="Optional custom tokenizer (HuggingFace ID).")
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
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--train_embeddings", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    args = parser.parse_args()

    hf_model_name = get_model_name(args.model)
    slug = model_slug(args.model)
    cache_dir = os.path.join(PROJECT_DIR, "cache", slug)
    os.makedirs(cache_dir, exist_ok=True)

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
            "--classifier", "transformer",
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
            "--patience", str(args.patience),
            "--label_smoothing", str(args.label_smoothing),
            "--grad_clip_norm", str(args.grad_clip_norm),
        ],
        3: [
            "--mode", "transformer",
        ],
    }
    if args.train_embeddings:
        phase_args[2].append("--train_embeddings")
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
    print("Transformer Fine-Tuning Pipeline")
    print("=" * 70)
    print(f"Model:         {hf_model_name}  (--model {args.model})")
    if args.tokenizer:
        print(f"Tokenizer:     {args.tokenizer}")
    print(f"Cache dir:     {cache_dir}")
    print(f"Max length:    {args.max_length}")
    print(f"Epochs:        {args.epochs}")
    print(f"Unfreeze last: {args.unfreeze_layers} layer(s)")
    print(f"Pooling:       {args.pooling}")
    print(f"Running phases: {[p[0] for p in phases]}")

    total_start = time.time()
    for num, script, desc in phases:
        if not run_phase(script, f"[{num}/{len(PHASES)}] {desc}", extra_env, phase_args[num]):
            print(f"\nPipeline stopped at phase {num}.")
            print(f"To resume: python run_transformer.py --start_phase {num}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    report_path = os.path.join(PROJECT_DIR, "results", "transformer_eval_report.txt")
    print(f"\n{'='*70}")
    print(f"All phases complete in {total_elapsed/60:.1f} minutes.")
    print(f"Results: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
