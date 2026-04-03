# Codex Journal

Purpose: persistent working notes for understanding how the upstream CBDC code maps onto this NLP project, with emphasis on unsupervised sentiment analysis and later DebiasVL integration.

## Scope

- Upstream reference code: `/Users/kimseunghyun/hsj_waterbird/direct_port`
- Current NLP project: `/Users/kimseunghyun/cs4248/cs4248-team-26/project`

## Session Notes

### 2026-04-03

- Created this journal so reasoning and codebase understanding survive context compaction.
- Confirmed the upstream CBDC codebase is centered around text-side latent PGD debiasing plus an optional image-side adaptation stage.
- Confirmed the local NLP project already has a `cbdc/` package and a pipeline structure, so the adaptation work is likely already underway here rather than starting from scratch.
- Completed a first codebase read-through of both the local NLP project and the upstream `direct_port` reference implementation.

## Local NLP Project Architecture

- `run_all.py` is the main orchestrator. It runs:
  1. `data/embed.py`
  2. `cbdc/refine.py`
  3. `pipeline/classify.py`
  4. `pipeline/evaluate.py`
- `run_all_debug.py` is the same flow with an in-process debug option for IDE stepping.
- The local pipeline currently evaluates four conditions:
  - `B1 (raw)`
  - `D1 (debias_vl)`
  - `D2 (CBDC)`
  - `D3 (debias_vl->CBDC)`

## Local Data Flow

- `dataset.py` prefers a local CSV in `project/data/` and preserves optional metadata such as:
  - `entity`
  - `cleaned_tokens`
  - `selected_text`
  - `time_of_tweet`
  - `age_of_user`
  - `country`
- `data/embed.py` caches:
  - normalized encoder embeddings
  - token ids and attention masks
  - raw texts
  - optional metadata fields
- This cached train split is reused later for topic mining, not just for embeddings.

## Local Encoder Adaptation

- `encoder.py` is the key CLIP-to-NLP translation layer.
- The backbone is fully frozen by default.
- CBDC training happens by:
  - extracting the hidden states before the last transformer block
  - injecting a perturbation only at the CLS position
  - running only the final transformer block as the trainable tail
- This is the NLP analogue of the upstream CLIP setup where most of the model is frozen and only the target layer is adapted.

## Local Prompt System

- `cbdc/prompts.py` separates the project into two prompt families:
  - `debias_vl` prompts
  - pure CBDC prompts
- For `debias_vl`, the project mines topic/style nuisance prompts from the actual training data when possible.
- The miner prefers:
  - balanced high-frequency `entity` values
  - phrase candidates from cleaned tokens
  - curated topic/style patterns matched against the dataset
- For CBDC itself, the local project uses fixed style-bias pairs such as:
  - question-mark vs statement
  - exclamation-heavy vs ordinary punctuation
  - very short vs longer detailed
  - laughter/emoticons vs none
- Important design split:
  - `D2` uses these fixed style pairs directly.
  - `D3` uses `debias_vl`-discovered anchors first, then runs CBDC training.

## Local DebiasVL Stage

- `cbdc/refine.py::discover_confound_map` is the local closed-form `debias_vl` stage.
- It builds:
  - `spurious_cb` from topic/style-only prompts
  - `candidate_cb` from sentiment x topic crossed prompts
  - `S_pairs` for same-sentiment, different-topic pairs
- Then it computes:
  - `P0` = complement of the spurious subspace
  - `M` from averaged pairwise structure over `S_pairs`
  - `P_debias = P0 @ (lambda * M + I)^-1`
- It then extracts top SVD directions from `I - P_debias` and turns those into bipolar anchor poles.

## Local CBDC Stage

- `cbdc/refine.py::text_iccv` is the local text-only CBDC loop.
- Core structure:
  - tokenize class prompt groups, target prompts, and keep prompts
  - freeze the last layer during PGD
  - run bipolar PGD on target prompts
  - unfreeze the last layer
  - update the last layer using `match_loss + ck_loss`
  - select the best checkpoint using a centroid-based validation macro-F1 selector
- This differs from the upstream vision setup in evaluation:
  - upstream selects on bias/fairness metrics like worst-group accuracy
  - local project selects on sentiment validation macro-F1

## Local Losses and Additions

- `losses.py` cleanly separates the three important terms:
  - `l_bias_contrastive`
  - `l_semantic_preservation`
  - `l_ck`
- The local project adds one notable new idea:
  - `sent_orthogonal_pgd`
- When enabled, the PGD gradient is projected orthogonal to the current sentiment prototype basis before the update step.
- This is not part of the original upstream code and is an explicit NLP-specific safeguard against confound directions drifting into sentiment space.

## Current Evaluation State

- `pipeline/classify.py` is still supervised linear-probe evaluation on the cached embeddings.
- `pipeline/evaluate.py` summarizes accuracy/F1 and also reports a simple direction interpretability diagnostic.
- The current saved `results/eval_report.txt` shows:
  - `D1` slightly above raw baseline on test accuracy/F1
  - `D2` slightly below raw
  - `D3` close to raw and slightly below `D1`
- This means the unsupervised representation learning story is partly present, but the final evaluation loop is still supervised.

## Upstream Reference Architecture

- `train_bafa.py` is the main entry point in the upstream code.
- `base.py` contains the main collaboration wrapper and training loops.
- `utils/txt_input.py` defines prompt banks for each dataset and is one of the most important reference files.
- `can/attack/simple_pgd.py` contains the bipolar latent PGD routines.
- `can/attack/debias_vl.py` contains the closed-form projection method used for `Orth-Cali` style debiasing and reused in the upstream project.

## Upstream DebiasVL Notes

- `debais_vl_s_c(opts)` hardcodes:
  - text descriptions
  - spurious prompts
  - candidate prompts
  - `S` and `B` pair lists
  for each dataset.
- `debias_vl()` computes:
  - `P0` from the spurious prompt subspace
  - `M` from pairwise semantic structure over `S`
  - `P = P0 @ inverse(1000 * M + I)`
- The local `discover_confound_map()` clearly generalizes this same idea, but replaces the hardcoded dataset prompts with dynamically mined topic/style prompts for text.

## Upstream-to-Local Mapping

- Upstream CLIP text encoder target layer:
  - local frozen transformer body + trainable final transformer block
- Upstream `target_text`, `keep_text`, class prompts:
  - local prompt banks in `cbdc/prompts.py`
- Upstream `debias_vl` hardcoded confounds:
  - local mined topic/style bank
- Upstream fairness-centric selection:
  - local sentiment-centric validation selector

## Important Conceptual Observation

- The local project is not a pure direct port.
- It is a hybrid:
  - faithful to the upstream CBDC text-stage mechanics
  - but more explicit and modular about prompt banks, cached artifacts, and condition isolation
  - with at least one added method contribution: sentiment-orthogonal PGD

## Important Research Observation

- The current project already supports unsupervised representation shaping in Phase 2.
- But the final task evaluation is still supervised because Phase 3 trains linear probes.
- If the research goal is truly "sentiment analysis from unsupervised training", then the likely next conceptual gap is replacing or complementing the linear probe with:
  - prototype-only zero-shot classification
  - nearest-centroid classification from prompt prototypes
  - or another label-free evaluation path

## Clarification: Probe vs Evaluation vs Training

- There are four separate things that are easy to conflate:
  1. representation training
  2. classifier fitting
  3. checkpoint/model selection
  4. final metric evaluation

- In this project:
  - Phase 2 mostly does label-free representation shaping using prompts and PGD.
  - Phase 2 currently also uses labels for checkpoint selection via `selector_f1`.
  - Phase 3 trains a supervised linear probe.
  - Phase 4 uses labels to report final accuracy/F1.

- A linear probe is not the only way to produce `negative / neutral / positive` outputs.
- The most direct no-probe alternative is zero-shot prototype classification:
  - encode the input text
  - compare it against the learned class prompt prototypes
  - predict the argmax similarity among negative/neutral/positive

- This is closer to CLIP-style zero-shot classification and closer to the CBDC paper's intended story than a supervised probe.

- So the strict framing options are:
  - "unsupervised representation learning + supervised probe evaluation"
  - or, if Phase 3 is removed, "unsupervised representation learning + zero-shot sentiment classification + labeled test evaluation"

## Prototype Evaluation Implementation Plan

Goal: add a no-probe evaluation path that predicts `negative / neutral / positive` directly from class prompt prototypes, without using any shared top-level `.pt` artifacts that can be overwritten across models or conditions.

Status note:

- A previous prototype-based implementation existed earlier in the project history but is no longer present in the current codebase.
- Reimplementation should therefore be treated as a fresh, from-scratch path rather than a patch on top of surviving legacy code.

## Implemented Prototype Artifact Semantics

The current implementation now builds `class_prompt_prototypes.pt` per condition, not as one shared global file.

### Artifact location

- All prototype files are model-scoped and condition-scoped.
- Path pattern:
  - `cache/<model-slug>/conditions/<condition-slug>/class_prompt_prototypes.pt`
- So there is no single shared top-level prototype `.pt` across all models or all conditions.

### How each condition's prototype `.pt` is built

- `B1 (raw)`:
  - built from the raw encoder
  - encode the sentiment class prompt groups
  - pool each prompt group into one prototype
  - save as raw class prototypes

- `D1 (debias_vl)`:
  - start from the same class prompt groups
  - encode them with the raw encoder
  - apply `P_debias` to those prompt embeddings
  - renormalize and pool into one prototype per class
  - save as debiased class prototypes

- `D2 (CBDC)`:
  - built from the CBDC-trained encoder after `text_iccv`
  - encode the class prompt groups with the trained encoder
  - pool into one prototype per class
  - save as condition-specific CBDC prototypes

- `D3 (debias_vl->CBDC)`:
  - built from the combined trained encoder after `text_iccv`
  - encode the class prompt groups with that trained encoder
  - pool into one prototype per class
  - save as condition-specific combined prototypes

### Prototype classification behavior

- The new prototype classifier does not fit a probe.
- For each condition:
  - load that condition's embeddings
  - load that condition's `class_prompt_prototypes.pt`
  - compute cosine-similarity logits
  - predict `argmax` over the 3 class prototypes

### Why this avoids the old collision problem

- The prototype artifact is not shared:
  - not shared across models because `CACHE_DIR` is model-specific
  - not shared across conditions because each condition has its own directory
- The prototype evaluation results are also separate:
  - `results_prototype.pt`
  - `eval_report_prototype.txt`

## Audit Notes

### Logic sanity audit summary

- Prototype evaluation is now condition-scoped and does not rely on a shared global prototype file.
- `run_all.py` and `run_all_debug.py` now reject the invalid combination:
  - `--classifier prototype --skip_cbdc`
- Reason:
  - prototype mode needs Phase 2 to materialize `class_prompt_prototypes.pt`.
- `submit_new.slurm` keeps the conda-init source line unchanged.
- `submit_new.slurm` now accepts:
  - `CLASSIFIER` with default `probe`
  - `RUN_MODEL` with default `roberta`
- No shell syntax problems were found in either launcher script.
- No Python syntax problems were found in the modified pipeline files.

### Commands used during the audit

These are the exact commands used:

```bash
python -m compileall run_all.py run_all_debug.py cbdc/refine.py pipeline/prototype_classify.py pipeline/evaluate.py
python run_all.py --help
python pipeline/prototype_classify.py --help
bash -n submit_new.slurm
bash -n submit_new.sh
python run_all.py --classifier prototype --skip_cbdc
```

Expected result of the last command:

```text
ERROR: --skip_cbdc cannot be used with --classifier prototype.
Prototype evaluation needs Phase 2 to materialize class_prompt_prototypes.pt.
```

### How to use

Local runs:

```bash
python run_all.py --model roberta --classifier probe
python run_all.py --model roberta --classifier prototype
python run_all.py --model roberta --classifier prototype --start_phase 2
python run_all.py --model roberta --classifier probe --only_phase 4
```

Debug runner:

```bash
python run_all_debug.py --model roberta --classifier prototype --inprocess
```

SLURM runs:

```bash
sbatch submit_new.slurm
sbatch --export=ALL,CLASSIFIER=prototype submit_new.slurm
sbatch --export=ALL,RUN_MODEL=bertweet,CLASSIFIER=prototype submit_new.slurm
sbatch --export=ALL,START_PHASE=2,CLASSIFIER=prototype submit_new.slurm
sbatch --export=ALL,ONLY_PHASE=4,CLASSIFIER=probe submit_new.slurm
```

### Important caveat

- Probe mode can skip Phase 2 if the user only wants baseline raw evaluation logic.
- Prototype mode cannot skip Phase 2 because the prototypes are created there.

### 1. Keep all prototype artifacts condition-scoped

- Use the existing condition directories from `pipeline/artifacts.py`.
- Continue saving `class_prompt_prototypes.pt` under each condition directory.
- Do not introduce any top-level shared file like `cache/class_prompt_prototypes.pt`.
- For new evaluation outputs, use separate filenames such as:
  - `prototype_results.pt`
  - `prototype_test_disagreements.csv`
  - `prototype_eval_report.txt`

### 2. Materialize prototypes for every condition

- `D2` and `D3` already save `class_prompt_prototypes.pt`.
- Add prototype materialization for:
  - `B1 (raw)`
  - `D1 (debias_vl)`
- For `B1`:
  - encode `cls_text_groups` with the raw encoder
  - pool each group into one sentiment prototype
- For `D1`:
  - encode `cls_text_groups` with the raw encoder
  - apply `P_debias` to the class prompt embeddings
  - renormalize and pool into one sentiment prototype per class
- Store each prototype bank in that condition's own artifact directory.

### 3. Add a dedicated prototype-classification phase

- Create a new evaluation path instead of overloading the current probe code.
- Best option:
  - add `pipeline/prototype_classify.py`
- This script should:
  - iterate over conditions
  - load each condition's split embeddings
  - load each condition's `class_prompt_prototypes.pt`
  - compute cosine-similarity logits
  - predict `argmax` over the 3 prototypes
  - compare predictions to labeled val/test sets
- Save outputs per condition and aggregate summary to a condition-safe results file.

### 4. Keep probe and prototype results separate

- Do not reuse `results.pt` for both evaluation modes.
- Use:
  - `results_probe.pt` for the current linear-probe path
  - `results_prototype.pt` for the no-probe path
- Likewise, keep report files separate:
  - `results/eval_report_probe.txt`
  - `results/eval_report_prototype.txt`

### 5. Wire the runner cleanly

- Extend `run_all.py` with an eval-mode flag, for example:
  - `--eval_mode probe`
  - `--eval_mode prototype`
  - `--eval_mode both`
- Default could be `both` if the project wants to report both stories.
- If `prototype` is selected:
  - skip `pipeline/classify.py`
  - run `pipeline/prototype_classify.py`
  - then run a matching prototype-aware report script or a unified evaluator that reads the correct results file.

### 6. Make the paper framing explicit

- Report two settings separately:
  - prompt/prototype zero-shot classification
  - supervised linear-probe evaluation
- This avoids mixing the "unsupervised training" story with the "supervised downstream probe" story.

### 7. Optional follow-up if strict unsupervised framing is needed

- The current Phase 2 checkpoint selector uses labeled validation F1.
- If needed later, add an alternative selector that is label-free or based on prompt-space diagnostics only.
- This is a second-stage improvement and does not block prototype evaluation.

## Upstream CBDC Notes

- `train_bafa.py` is the main entry point and alternates text and image stages.
- `base.py` contains the main collaboration wrapper, dataset prompt setup, training loops, evaluation, and target-layer wrapper logic.
- `can/attack/simple_pgd.py` contains the latent PGD routines that generate paired perturbations and the image-side attack.
- `utils/txt_input.py` defines the prompt schema for class prompts, bias prompts, and mixed prompts. This is one of the most important files for any NLP transfer.
- `can/attack/debias_vl.py` is the DebiasVL-related component that still needs a dedicated read-through.

## Local NLP Project Notes

- Top-level structure includes `encoder.py`, `dataset.py`, `losses.py`, `config.py`, `run_all.py`, `pipeline/`, and `cbdc/`.
- The existence of `cbdc/prompts.py` and `cbdc/refine.py` suggests the project already has a text-native reinterpretation of the upstream method.

## Open Questions

- How closely does the local `cbdc/` package follow the upstream text-stage objective versus a simplified reinterpretation?
- Where are pseudo-labels or unsupervised topic/style biases mined in the local project?
- How is evaluation currently defined for the sentiment task: prototype similarity, linear head, or another classifier?
- How much of DebiasVL is conceptually needed for the NLP story versus only CBDC itself?
