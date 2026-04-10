"""
Minimal runner for the `vlm-personalization-pgd` branch.

This script focuses on a clean personalization setup:

1. load a frozen transformer with `encoder.py`
2. adapt instance/class media into dataloaders
3. build subject-centered prompt banks
4. run prompt discovery and/or ICCV-style text/image steering

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
    parser = argparse.ArgumentParser(description="Run PGD-based personalization setup and ICCV-style steering.")
    parser.add_argument("--instance_dir", required=True, help="Directory of subject instance images or frame folders.")
    parser.add_argument("--class_dir", default=None, help="Optional class-prior directory.")
    parser.add_argument("--concept_token", required=True, help="Personalized concept token, e.g. 'sks dog'.")
    parser.add_argument("--class_name", required=True, help="Base class phrase, e.g. 'a dog'.")
    parser.add_argument("--model", default="qwen25-3b", help="Backbone model key or HF model ID.")
    parser.add_argument(
        "--image_model",
        default=None,
        help="Optional VLM / CLIP model for `img_iccv`. Defaults to `--model` if omitted.",
    )
    parser.add_argument("--tokenizer", default=None, help="Optional custom tokenizer.")
    parser.add_argument(
        "--mode",
        default="discover",
        choices=["discover", "text_iccv", "img_iccv", "iccv", "all"],
        help="`discover` keeps the old PGD mining path, `iccv` runs text+img steering, `all` runs everything.",
    )
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
    parser.add_argument("--text_iters", type=int, default=20)
    parser.add_argument("--img_iters", type=int, default=5)
    parser.add_argument("--text_lr", type=float, default=1e-3)
    parser.add_argument("--img_lr", type=float, default=1e-3)
    parser.add_argument("--img_bound", type=float, default=0.1)
    parser.add_argument("--img_step", type=float, default=1e-2)
    parser.add_argument("--img_attack_steps", type=int, default=10)
    parser.add_argument("--img_loss_scale", type=float, default=100.0)
    parser.add_argument("--img_lambda", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PersonalizationConfig(
        model=args.model,
        image_model=args.image_model,
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
        text_iters=args.text_iters,
        img_iters=args.img_iters,
        text_lr=args.text_lr,
        img_lr=args.img_lr,
        img_bound=args.img_bound,
        img_step=args.img_step,
        img_attack_steps=args.img_attack_steps,
        img_loss_scale=args.img_loss_scale,
        img_lambda=args.img_lambda,
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
        text_slug = model_slug(args.model)
        image_slug = model_slug(args.image_model or args.model)
        if image_slug != text_slug:
            slug = f"{text_slug}__img__{image_slug}"
        else:
            slug = text_slug
        concept_slug = args.concept_token.strip().replace(" ", "_").replace("/", "_")
        output_dir = os.path.join(PROJECT_DIR, "cache", slug, "personalization", concept_slug)

    run_discovery = args.mode in {"discover", "all"} and not args.skip_discovery
    run_text_iccv = args.mode in {"text_iccv", "iccv", "all"}
    run_img_iccv = args.mode in {"img_iccv", "iccv", "all"}

    discovery = None
    if run_discovery:
        discovery = trainer.discover(prompt_bank)

    text_iccv_artifacts = None
    if run_text_iccv:
        text_iccv_artifacts = trainer.text_iccv(prompt_bank)

    img_iccv_artifacts = None
    if run_img_iccv:
        img_iccv_artifacts = trainer.img_iccv(
            prompt_bank=prompt_bank,
            data_payload=data_payload,
            text_artifacts=text_iccv_artifacts,
        )

    trainer.save_run(
        output_dir=output_dir,
        prompt_bank=prompt_bank,
        data_payload=data_payload,
        discovery=discovery,
        text_iccv_artifacts=text_iccv_artifacts,
        img_iccv_artifacts=img_iccv_artifacts,
    )

    print("=" * 72)
    print("Personalization Setup Complete")
    print("=" * 72)
    print(f"Model:           {get_model_name(args.model)}")
    print(f"Mode:            {args.mode}")
    print(f"Media mode:      {args.media_mode}")
    print(f"Instance items:  {len(data_payload['instance_records'])}")
    print(f"Class items:     {len(data_payload['class_records'])}")
    print(f"Output dir:      {output_dir}")
    if discovery is not None:
        print(f"Directions:      {tuple(discovery.directions.shape)}")
        print(f"Axes:            {', '.join(discovery.axis_names)}")
        print(f"Top singulars:   {discovery.singular_values}")
    if text_iccv_artifacts is not None:
        last_text = text_iccv_artifacts.losses[-1]
        print(f"Text ICCV:       loss={last_text['loss']:.4f} dirs={tuple(text_iccv_artifacts.directions.shape)}")
    if img_iccv_artifacts is not None:
        last_img = img_iccv_artifacts.losses[-1]
        print(f"Img ICCV:        loss={last_img['loss']:.4f} model={trainer.image_model_name}")


if __name__ == "__main__":
    main()
