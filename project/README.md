# VLM Personalization PGD

This branch has been cleaned down to the pieces that are still useful for the
new direction:

- frozen transformer loading in `encoder.py`
- generic PGD losses in `losses.py`
- model lookup helpers in `config.py`
- personalization-first data, prompts, and PGD discovery in `personalization/`
- a dedicated runner in `run_personalization.py`

The old `D1 / D2 / D3` sentiment-condition pipeline has been removed.

## Core Entry Point

Run the personalization scaffold directly:

```bash
python run_personalization.py \
  --instance_dir data/my_subject/instance \
  --class_dir data/my_subject/class \
  --concept_token "sks dog" \
  --class_name "a dog" \
  --mode discover
```

## Available Modes

`run_personalization.py` now has three stage families:

- `--mode discover`
  Prompt-only nuisance mining with the salvaged `encoder.py` path
- `--mode text_iccv`
  ICCV-style text steering on frozen text embeddings plus a trainable target head
- `--mode img_iccv`
  ICCV-style image steering on frozen VLM / CLIP image features plus a trainable target head
- `--mode iccv`
  Runs `text_iccv` then `img_iccv`
- `--mode all`
  Runs discovery plus both ICCV stages

Artifacts are written under:

```text
cache/<model-slug>/personalization/<concept-slug>/
```

## Expected Data Layout

Image mode:

```text
path/to/instance/
  img_000.png
  img_001.png

path/to/class/
  cls_000.png
  cls_001.png
```

Video-frame mode:

```text
path/to/instance/
  clip_a/
    0001.png
    0002.png
  clip_b/
    0001.png
```

Use `--media_mode video_frames` for the second layout.

## Main Files

- `encoder.py`
  Reusable frozen transformer loader plus latent-tail perturbation path
- `losses.py`
  Generic contrastive and semantic-preservation losses
- `personalization/iccv.py`
  ICCV-style text/image steering loops and VLM / CLIP embedding wrapper
- `personalization/data.py`
  Adapted instance/class media loader
- `personalization/prompts.py`
  Subject-centered prompt bank construction
- `personalization/pgd.py`
  Bipolar PGD nuisance-direction discovery
- `personalization/trainer.py`
  Minimal orchestration and artifact writing
- `personalization/projection.py`
  Generic projection utility

## Launcher Scripts

The shell and SLURM launchers now wrap `run_personalization.py` instead of the
removed sentiment pipeline:

- `submit_new.sh`
- `submit_new.slurm`

They expect environment variables such as:

- `INSTANCE_DIR`
- `CLASS_DIR` (optional)
- `CONCEPT_TOKEN`
- `CLASS_NAME`
- `RUN_MODEL`
- `RUN_MODE`
- `IMAGE_MODEL` (optional for `img_iccv`)

## Current Scope

This repo is now an image-first, video-next personalization scaffold.
Qwen, Gemma, BERT-family, and CLIP-style backbones can now be used in two ways:

- text-side discovery / `text_iccv` through frozen text embeddings
- image-side `img_iccv` through multimodal `get_image_features` or CLIP fallback

The current implementation is a steering scaffold, not a generator trainer.

## Example Commands

Prompt-only discovery:

```bash
python run_personalization.py \
  --instance_dir data/my_subject/instance \
  --class_dir data/my_subject/class \
  --concept_token "sks dog" \
  --class_name "a dog" \
  --model bert \
  --mode discover
```

Text-side ICCV steering:

```bash
python run_personalization.py \
  --instance_dir data/my_subject/instance \
  --class_dir data/my_subject/class \
  --concept_token "sks dog" \
  --class_name "a dog" \
  --model bert \
  --mode text_iccv
```

Image-side ICCV steering with a VLM or CLIP-style backend:

```bash
python run_personalization.py \
  --instance_dir data/my_subject/instance \
  --class_dir data/my_subject/class \
  --concept_token "sks dog" \
  --class_name "a dog" \
  --model bert \
  --image_model Qwen/Qwen2.5-VL-3B-Instruct \
  --mode img_iccv
```
