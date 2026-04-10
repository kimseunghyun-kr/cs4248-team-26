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
- primarily through prompt-conditioned feature steering
- not LoRA-first or DreamBooth-first fine-tuning

In practice that means:

- a VLM or transformer encoder provides concept grounding, feature detection, and identity preservation signals
- CBDC-style PGD discovers steerable latent directions
- later, a generator can consume those steered conditioning representations if needed

## Proposed Goal

Goal:

- personalization for a subject or concept through prompt-conditioned feature control
- PGD is used to discover nuisance or attribute directions
- those directions can then be used to either amplify a desired feature or suppress an undesired bias
- text-to-image first
- text-to-video second

Core idea:

- use a VLM or transformer embedding space to discover what should change and what should stay invariant
- use PGD to find hard nuisance directions such as background, viewpoint, pose, lighting, or style leakage
- keep subject identity stable while allowing controlled movement along chosen feature axes

## Mapping The Current CBDC Design To Personalization

Current CBDC component -> personalization analogue

- `cls_text_groups` -> concept identity prompt set and class-prior prompt set
- `target_text` -> personalization prompts for the subject or concept token
- `keep_text` -> preservation prompts that should keep identity and broad semantics stable
- `bias_anchors` / `anti_anchors` -> nuisance poles such as indoor vs outdoor, close-up vs full-body, realistic vs stylized
- `sent_orthogonal_pgd` -> identity-orthogonal PGD so nuisance discovery does not collapse the subject embedding itself
- `P_debias` or discovered direction bank -> steering operator for removal or amplification
- validation macro-F1 selector -> feature detectability, identity stability, and later generation quality if a generator is added

## Recommended MVP Stack

Phase A: image first

1. Use one VLM wrapper:
   - QwenVL or Gemma-VL
2. Run PGD in a VLM-conditioned space:
   - discover nuisance directions from the personalization set and class-prior set
3. Expose a steering interface:
   - detect feature score from a prompt
   - remove the feature if it is a bias
   - amplify the feature if it is desired
4. Evaluate:
   - identity consistency
   - prompt following
   - background or style disentanglement
   - controllability of attribute steering

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
4. Add prompt-side feature scoring and steering operators.
5. Only after the steering path is stable, consider downstream generator integration.
6. Extend the same idea to video-aware axes later.

## Practical Decision

The safest next code move is not to stretch the current sentiment pipeline into video or image generation directly.

The right move is to keep the reusable optimization ideas, then build a new personalization module beside the existing pipeline.

## Journal Notes

### 2026-04-11 — How CBDC-style PGD maps to personalization

Core question:

- In the original CBDC code, PGD is used to discover cleaner bias directions in embedding space.
- In this branch, the analogous goal is to discover nuisance directions for a personalized subject or concept before or during generator training.

The clean personalization story is:

1. Choose a personalized concept token.
2. Collect a few instance images or frames of that concept.
3. Define what should stay stable:
   - subject identity
   - broad semantic meaning
4. Define what should be allowed to vary:
   - background
   - framing
   - lighting
   - style
   - later, motion and camera behavior for video
5. Use PGD to push prompt-conditioned embeddings toward opposite nuisance poles while explicitly preserving identity.
6. Compress the discovered perturbation differences into a small nuisance-direction bank.
7. Use those directions as steering operators:
   - remove a bias
   - amplify a desired feature
   - or detect whether a prompt strongly activates that feature

### Current branch interpretation

The current branch uses the same structural idea as CBDC, but in a stripped-down form:

- frozen backbone encoder
- prompt-conditioned latent PGD
- semantic preservation loss
- low-rank nuisance direction discovery through SVD

Important clarification:

- the point of this branch is **not** LoRA fine-tuning
- the point is personalization through controllable feature steering
- CBDC-style PGD is the mechanism for discovering those steerable directions

So the branch is currently implementing the discovery half of the method:

- build the subject-conditioned prompt geometry
- discover nuisance directions with PGD
- save those directions as artifacts for later steering

### How the current code actually works

The relevant files are:

- [personalization/prompts.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/personalization/prompts.py)
- [personalization/pgd.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/personalization/pgd.py)
- [personalization/trainer.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/personalization/trainer.py)
- [encoder.py](/Users/kimseunghyun/cs4248/cs4248-team-26/project/encoder.py)

The implementation flow is:

1. Build a prompt bank.

Current prompt roles:

- `identity_prompts`
  prompts that define the subject identity, for example:
  - `a photo of sks dog`
  - `a portrait of sks dog`
- `class_prompts`
  generic class-prior prompts, for example:
  - `a photo of a dog`
- `keep_prompts`
  prompts used to preserve broad semantic faithfulness, for example:
  - `a faithful photo of sks dog`
- `target_prompts`
  prompts that are actively perturbed during PGD, for example:
  - `a cinematic photo of sks dog`
- `nuisance_axes`
  opposite poles such as:
  - indoor vs outdoor
  - close-up vs full-body
  - realistic vs stylized
  - daylight vs night

2. Turn nuisance axes into anchor banks.

For each nuisance axis, the code encodes:

- positive prompts
- negative prompts

and averages them into:

- `bias_anchors`
- `anti_anchors`

These are the personalization analogue of CBDC’s bipolar prompt poles.

3. Build an identity basis.

The code encodes `identity_prompts` and uses them as the subject-preservation basis.

This is the most important adaptation of the old sentiment-orthogonal idea:

- in the sentiment project, the PGD step was projected away from sentiment prototypes
- in this branch, the PGD step is projected away from identity prompts

That means the discovered nuisance directions are encouraged to vary the subject’s context, not erase the subject itself.

4. Build the keep bank.

The code encodes `keep_prompts` and uses them in the semantic preservation loss:

- if PGD drifts too far from the original target embedding along these semantic axes, the loss pushes back

5. Run bipolar PGD on the target prompts.

For each `target_prompt`:

- get the intermediate hidden representation from the frozen backbone
- inject a latent delta
- run the trainable tail path
- get the perturbed embedding

Then for each restart, the code runs two PGD branches:

- one branch pushes toward the positive nuisance poles
- one branch pushes toward the negative nuisance poles

The objective is:

- `l_bias_contrastive`
  push the perturbed embedding toward one side of each nuisance axis
- `l_semantic_preservation`
  keep the perturbed embedding near the original semantic content

And the gradient is projected orthogonal to the identity basis before the PGD update.

This is the current branch’s cleanest “CBDC for personalization” idea.

6. Form a nuisance direction bank.

After PGD finishes:

- collect all positive perturbed embeddings
- collect all negative perturbed embeddings
- compute `delta_bank = z_adv_pos - z_adv_neg`

This gives a set of discovered variation directions that are:

- prompt-conditioned
- nuisance-seeking
- identity-constrained

7. Compress with SVD.

The code centers the delta bank and runs SVD.

The top right-singular vectors become:

- `pgd_directions.pt`

These are the low-rank nuisance directions currently saved for each run.

### Steering view: remove vs amplify

This branch should be thought of as a steering system.

Let:

- `z` be a prompt embedding
- `d` be one discovered nuisance direction
- `U` be a discovered nuisance basis with multiple rows

Then the most natural uses are:

Single-direction detection:

- feature score = `z · d`

Single-direction removal:

- `z_remove = normalize(z - alpha * (z · d) * d)`

Single-direction amplification:

- `z_amplify = normalize(z + beta * (z · d) * d)`

Subspace removal:

- `z_remove = normalize(z - (z U^T) U)`

Subspace amplification:

- `z_amplify = normalize(z + gamma * (z U^T) U)`

So the useful interpretation is not necessarily:

- literal inversion of a full CBDC matrix

but rather:

- detect the confound component
- subtract it if it is bias
- add more of it if it is the desired personalized feature

If a full debiasing matrix `P` is available, then:

- removal is `P z`
- the removed component is approximately `(I - P) z`
- amplification can be framed as adding back a scaled amount of that removed component

That is the most practical “invert or amplify” interpretation for this branch.

### What personalization means here

In personalization terms, the problem is:

- learn the subject token
- preserve who or what the subject is
- avoid overfitting to accidental traits in the few-shot images

The accidental traits are exactly where CBDC-style PGD is useful.

Example:

- all your subject photos are outdoors and close-up
- a naive personalization method may entangle:
  - subject identity
  - outdoor background
  - close-up framing

The PGD discovery stage tries to explicitly surface those nuisance directions so a later steering stage can say:

- keep the subject
- do not tie the subject too strongly to outdoor scenes
- do not tie the subject too strongly to close-up shots
- or, if desired, intentionally amplify one of those features

### What is implemented vs not yet implemented

Implemented now:

- instance/class media loading
- subject-conditioned prompt bank construction
- prompt-only nuisance-anchor discovery
- identity-orthogonal bipolar PGD
- SVD compression of nuisance directions
- artifact saving for later use
- the mathematical ingredients needed for feature detection and removal/amplification

Not implemented yet:

- a user-facing steering API that takes a prompt and applies remove/amplify operations directly
- using the instance or class images directly inside the PGD objective
- image-conditioned or video-conditioned VLM scoring
- temporal nuisance discovery for real video generation
- downstream generator integration if generation is desired later

Important current limitation:

- `class_dir` is loaded and recorded now, but the current PGD discovery loop is still prompt-driven rather than image-driven
- so class images are staged for future personalization supervision, not yet used as active supervision inside `personalization/pgd.py`

### Intended next steering step

The most natural next step is:

1. keep the current PGD discovery stage
2. add a prompt-side feature scoring API
3. expose remove/amplify operations on the discovered direction bank
4. only later decide whether a generator should consume those steered embeddings

Concretely, the next branch utility should likely include:

- prompt embedding extraction
- feature score reporting per discovered axis
- signed steering strength control
- save-before/save-after comparison artifacts for prompt embeddings

So the big picture is:

- current branch = discover the nuisance geometry
- next branch step = use that geometry to steer prompts and features directly

## 2026-04-11 - Implemented `text_iccv` and `img_iccv`

This branch now has both upstream-inspired ICCV loops implemented in code.

### `text_iccv`

Implemented in `personalization/iccv.py`.

Current branch-local interpretation:

- freeze the text encoder feature source
- extract prompt embeddings for:
  - class prompts
  - keep prompts
  - target prompts
  - nuisance-axis positive / negative prompts
- learn a small target adapter on top of those frozen embeddings
- run bipolar PGD on the target prompt embeddings
- compute `S = adv_pos - adv_neg`
- minimize:
  - match loss: keep `S` orthogonal to class semantics
  - `L_ck`: keep nuisance-pair differences orthogonal to class semantics

This is not a LoRA path and not a generator path.
It is a steering-space path.

### `img_iccv`

Also implemented in `personalization/iccv.py`.

Current branch-local interpretation:

- load a VLM image encoder through HF `get_image_features`
- fallback to CLIP-style `get_image_features` if needed for testing
- freeze the image feature source
- learn a small target adapter in image-embedding space
- generate adversarial image-feature perturbations with the BAFA / CBDC-style image PGD objective
- train the image adapter so the perturbation difference becomes less aligned with the debiased class text anchors

Important detail:

- for Qwen2.5-VL / Gemma 4, this is using the model's multimodal projected image features
- so the branch now has a real image-side steering path, not only prompt-only mining

### Why this structure

This is the cleanest version that fits the stripped branch:

- keep `encoder.py` for salvaged text-side loading and discovery
- add a shared ICCV steering module
- use the real multimodal feature path for images
- avoid dragging the old D1 / D2 / D3 training stack back in

So the branch is now:

- discovery
- text-side steering
- image-side steering

and all three stages save artifacts separately.
