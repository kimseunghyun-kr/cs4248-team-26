"""
Minimal runner for the `vlm-personalization-pgd` branch.

This script focuses on a clean personalization setup:

1. load a frozen transformer with `encoder.py`
2. adapt instance/class media into dataloaders
3. build subject-centered prompt banks
4. run upstream-style PGD nuisance discovery

Example:
  python run_personalization.py \
      --instance_dir data/my_subject/instance \
      --class_dir data/my_subject/class \
      --concept_token "sks dog" \
      --class_name "a dog"
"""

from __future__ import annotations

import argparse
import os

import torch

from config import get_model_name, model_slug
from personalization import PersonalizationConfig, PersonalizationTrainer


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PGD-based personalization setup and discovery.")
    parser.add_argument("--instance_dir", required=True, help="Directory of subject instance images or frame folders.")
    parser.add_argument("--class_dir", default=None, help="Optional class-prior directory.")
    parser.add_argument("--concept_token", required=True, help="Personalized concept token, e.g. 'sks dog'.")
    parser.add_argument("--class_name", required=True, help="Base class phrase, e.g. 'a dog'.")
    parser.add_argument("--model", default="qwen25-3b", help="Backbone model key or HF model ID.")
    parser.add_argument("--tokenizer", default=None, help="Optional custom tokenizer.")
    parser.add_argument("--media_mode", default="image", choices=["image", "video_frames"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--skip_discovery", action="store_true", help="Only validate data/prompts; skip PGD discovery.")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--n_pgd_steps", type=int, default=20)
    parser.add_argument("--step_lr", type=float, default=0.0037)
    parser.add_argument("--keep_weight", type=float, default=0.92)
    parser.add_argument("--num_restarts", type=int, default=8)
    parser.add_argument("--random_eps", type=float, default=0.22)
    parser.add_argument("--n_bias_dirs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PersonalizationConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        media_mode=args.media_mode,
        epsilon=args.epsilon,
        n_pgd_steps=args.n_pgd_steps,
        step_lr=args.step_lr,
        keep_weight=args.keep_weight,
        num_restarts=args.num_restarts,
        random_eps=args.random_eps,
        n_bias_dirs=args.n_bias_dirs,
        device=device,
    )

    trainer = PersonalizationTrainer(cfg)
    prompt_bank = trainer.build_prompt_bank(
        concept_token=args.concept_token,
        class_name=args.class_name,
    )
    data_payload = trainer.build_data(
        instance_dir=args.instance_dir,
        class_dir=args.class_dir,
        prompt_bank=prompt_bank,
    )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        slug = model_slug(args.model)
        concept_slug = args.concept_token.strip().replace(" ", "_").replace("/", "_")
        output_dir = os.path.join(PROJECT_DIR, "cache", slug, "personalization", concept_slug)

    discovery = None
    if not args.skip_discovery:
        discovery = trainer.discover(prompt_bank)

    trainer.save_run(
        output_dir=output_dir,
        prompt_bank=prompt_bank,
        data_payload=data_payload,
        discovery=discovery,
    )

    print("=" * 72)
    print("Personalization Setup Complete")
    print("=" * 72)
    print(f"Model:           {get_model_name(args.model)}")
    print(f"Media mode:      {args.media_mode}")
    print(f"Instance items:  {len(data_payload['instance_records'])}")
    print(f"Class items:     {len(data_payload['class_records'])}")
    print(f"Output dir:      {output_dir}")
    if discovery is not None:
        print(f"Directions:      {tuple(discovery.directions.shape)}")
        print(f"Axes:            {', '.join(discovery.axis_names)}")
        print(f"Top singulars:   {discovery.singular_values}")


if __name__ == "__main__":
    main()
