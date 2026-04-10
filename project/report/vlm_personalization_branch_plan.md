# VLM Personalization Branch Plan

Status:

- Branch created: `vlm-personalization-pgd`
- Base commit: `a59be2f`
- Intention: start a new line of work for VLM-guided personalization for text-to-image first, then text-to-video

## Implemented Scaffold

The branch now includes a clean personalization-first path:

- [run_personalization.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/run_personalization.py)
  New runner for concept setup, data adaptation, and PGD nuisance discovery.
- [personalization/data.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/personalization/data.py)
  Instance/class media loader for images now and frame-folder videos later.
- [personalization/prompts.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/personalization/prompts.py)
  Identity, class-prior, keep, target, and nuisance-axis prompt bank construction.
- [personalization/pgd.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/personalization/pgd.py)
  Upstream-style bipolar PGD discovery with identity-preserving gradient projection.
- [personalization/trainer.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/personalization/trainer.py)
  Minimal trainer/orchestrator around the kept transformer loader in [encoder.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/encoder.py).
- [personalization/projection.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/personalization/projection.py)
  Salvaged generic projection helper kept outside the old pipeline namespace.

## What This Repo Already Gives Us

These are the pieces that survived the cleanup:

- [encoder.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/encoder.py)
  Frozen-backbone plus trainable-tail adaptation, with latent perturbation support.
- [losses.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/losses.py)
  Clean separation of bias-contrastive, semantic-preservation, and cross-knowledge losses.
- [config.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/config.py)
  Shared model-registry and model-slug helpers.

The upstream reference in `/Users/kimseunghyun/hsj_waterbird/direct_port` confirms the same core pattern:

- frozen multimodal backbone
- adversarial direction discovery with PGD
- small trainable target module
- text step and optional image step

## What Does Not Transfer Directly

The earlier sentiment-condition stack has been removed. What remains is the reusable optimization machinery plus the new personalization path.

The current latent injection path in [encoder.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/encoder.py)
  It is built around the last text block and CLS or final-token pooling, not around diffusion or video generation modules.

## Important Architectural Clarification

QwenVL and Gemma-VL are useful as multimodal encoders, judges, retrievers, and prompt-conditioned concept checkers.

They are not, by themselves, DreamBooth-style text-to-image or text-to-video generators.

So the clean framing for this branch should be:

- VLM-guided personalization
- not VLM-only generation

In practice that means:

- a VLM provides concept grounding, hard-negative mining, semantic checks, and identity preservation signals
- a separate generator backbone handles image or video synthesis

## Proposed Goal

Goal:

- DreamBooth-like personalization for a subject or concept
- PGD is used to discover and regularize nuisance directions
- text-to-image first
- text-to-video second

Core idea:

- personalize a generator on a small subject dataset
- use a VLM embedding space to discover what should change and what should stay invariant
- use PGD to find hard nuisance directions such as background, viewpoint, pose, lighting, or style leakage
- constrain training so the personalized concept stays identifiable while nuisance entanglement is reduced

## Mapping The Current CBDC Design To Personalization

Current CBDC component -> personalization analogue

- `cls_text_groups` -> concept identity prompt set and class-prior prompt set
- `target_text` -> personalization prompts for the subject or concept token
- `keep_text` -> preservation prompts that should keep identity and broad semantics stable
- `bias_anchors` / `anti_anchors` -> nuisance poles such as indoor vs outdoor, close-up vs full-body, realistic vs stylized
- `sent_orthogonal_pgd` -> identity-orthogonal PGD so nuisance discovery does not collapse the subject embedding itself
- validation macro-F1 selector -> VLM similarity, prior-preservation loss, reconstruction quality, and later human preference or retrieval metrics

## Recommended MVP Stack

Phase A: image first

1. Use one VLM wrapper:
   - QwenVL or Gemma-VL
2. Use one image generator:
   - a diffusion or DiT backbone with LoRA support
3. Keep adaptation light:
   - LoRA or small adapters first
4. Run PGD in a VLM-conditioned space:
   - discover nuisance directions from the personalization set and class-prior set
5. Evaluate:
   - identity consistency
   - prompt following
   - background or style disentanglement

Phase B: video second

1. Reuse the same subject identity bank
2. Add temporal-consistency losses
3. Extend nuisance discovery to motion, camera angle, and frame-to-frame drift

## Suggested Repository Split For This Branch

Not implemented yet, but this is the cleanest next layout:

- `personalization/datasets/`
  small concept dataset loader for image or video personalization
- `personalization/vlm/`
  QwenVL and Gemma-VL wrappers for embeddings, scoring, and prompt-conditioned checks
- `personalization/pgd/`
  generic bipolar PGD and nuisance-direction discovery
- `personalization/generators/`
  image and later video generator adapters
- `personalization/train/`
  DreamBooth-like or LoRA fine-tuning loops
- `personalization/eval/`
  VLM similarity, identity retrieval, and generation-quality reporting

## First Implementation Order

1. Generalize the current PGD code out of sentiment-specific prompt banks.
2. Add a concept dataset format for subject images and optional subject videos.
3. Add a VLM wrapper layer for embedding extraction and concept scoring.
4. Add a generator wrapper for one text-to-image backbone.
5. Build a small PGD-guided personalization loop for image generation.
6. Only after the image path is stable, extend to video.

## Practical Decision

The safest next code move is not to stretch the current sentiment pipeline into video or image generation directly.

The right move is to keep the reusable optimization ideas, then build a new personalization module beside the existing pipeline.
