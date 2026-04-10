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
  --class_name "a dog"
```

## What The Current Runner Does

`run_personalization.py` performs four things:

1. Loads a frozen backbone through `encoder.py`
2. Adapts instance and optional class-prior media into loaders
3. Builds identity, keep, target, and nuisance prompt banks
4. Runs upstream-style bipolar PGD nuisance discovery

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

## Current Scope

This repo is now an image-first, video-next personalization scaffold.
Qwen, Gemma, Llama, BERT-family, and related backbones can be used as frozen
text-side encoders or latent-tail adaptation targets, but actual DreamBooth-like
generation still requires a separate generator backbone to be added on top.
tries to materialize them from the Phase 2 artifacts automatically.
