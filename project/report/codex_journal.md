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

## 2026-04-04 Prototype Results And Prompt Ratification

### Prototype evaluation outcome

Prototype evaluation on `roberta` showed:

- `B1 (raw)`: test acc `0.4110`, test F1 `0.3535`
- `D1 (debias_vl)`: test acc `0.4541`, test F1 `0.4418`
- `D2 (CBDC)`: test acc `0.4442`, test F1 `0.4451`
- `D3 (debias_vl->CBDC)`: test acc `0.4231`, test F1 `0.3615`

Interpretation:

- `D1` and `D2` clearly improve over raw under prototype evaluation.
- This supports the claim that the prompt/prototype space is being improved by the unsupervised alignment method, even when the downstream supervised probe story is weaker.
- `D3` is still weaker than `D1` and `D2`, so the combined path is not yet the best variant.

### Ratification status for `cbdc/prompts.py`

Current prompt design is partially ratified by the dataset analysis.

Strongly supported:

- `class = sentiment`
  - negative / neutral / positive as in `make_cls_text_groups()`
- topic-oriented nuisance prompts for `debias_vl`
  - e.g. `today`, `work`, `sleep`, `friends`, `food`, `school`, `twitter`
- some style shortcut prompts as candidate nuisance axes
  - question marks
  - repeated question marks
  - URL/link
  - very short text

Supported but risky:

- emoticons
- internet laughter (`lol`, `haha`)
- exclamation-heavy style

Reason:

- the data analysis shows these are correlated with sentiment
- but they may also carry genuine sentiment signal, not just nuisance signal
- so they are useful shortcut probes, but not obviously “pure bias” axes

Not ratified as bias anchors:

- direct sentiment lexemes such as:
  - `love`, `good`, `happy`, `thanks`, `sad`, `hate`, `awesome`, `bad`
- metadata like:
  - `Time of Tweet`
  - `Age of User`

Reason:

- sentiment lexemes are class signal, not nuisance signal
- time/age appear nearly sentiment-balanced in the analysis

### Data-analysis-based rationale to preserve

From the project’s exploratory analysis:

- random-forest / logistic features show clear sentiment-bearing lexemes:
  - positive: `love`, `awesome`, `thanks`, `happy`, `great`, `good`
  - negative: `sad`, `miss`, `sorry`, `sucks`, `hate`, `bad`
- style confounds are real and skewed:
  - question marks
  - multiple questions
  - URL/link
  - very short texts
  - emoticons
  - laughter markers
- `selected_text` is structurally dangerous:
  - neutral tweets select almost the full tweet
  - positive/negative tweets select only short spans
- this strongly supports the current choice to avoid using `selected_text` as direct supervised signal for prompt construction and instead only use it as something to subtract from mining context

### Prompt-ratification plan

If prompt revisions are made later, the safest order is:

1. Keep class prompts as sentiment-semantic prompts.
2. Keep topic-based `debias_vl` anchors as the primary nuisance bank.
3. Keep URL / question-mark / short-text style anchors.
4. Move emoticon / laughter / exclamation anchors into:
   - ablation sets
   - secondary style-anchor banks
   - or lower-weighted perturbation banks
5. Continue excluding obvious class words from bias-anchor banks.

## Research-Backed Measures For Better Anchors And Prompts

### High-confidence resources

- VADER:
  - Hutto and Gilbert, 2014
  - official paper page: https://ocs.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/viewPaper/8109
- MPQA / contextual polarity:
  - Wilson, Wiebe, and Hoffmann, 2009
  - https://aclanthology.org/J09-3003/
- NRC Emotion Lexicon:
  - Mohammad and Turney, 2013
  - official resource page: https://www.saifmohammad.com/WebPages/AccessResource.htm
- NRC VAD Lexicon:
  - official page: https://saifmohammad.com/WebPages/nrc-vad.html
- NRC Affect / Emotion Intensity Lexicon:
  - official page: https://www.saifmohammad.com/WebPages/AffectIntensity.htm
- SentiWordNet:
  - Esuli and Sebastiani, 2006
  - https://aclanthology.org/L06-1225/
- NRC-Canada Twitter sentiment features:
  - official summary page: https://saifmohammad.com/WebPages/NRC-Canada-Sentiment.htm
  - system paper: https://aclanthology.org/S13-2053.pdf

### Proposed measures for class prompts

1. Lexicon-backed class prompt expansion

- Build prompt paraphrases from words that are:
  - high-valence positive / low-valence negative in NRC VAD
  - high intensity in NRC Affect Intensity
  - strong subjective / stable polarity in MPQA or contextual-polarity resources
- Keep neutral prompts explicitly low-affect and objective.

2. Prompt-bank coherence score

- For each class prompt group:
  - compute average within-group cosine similarity
  - compute average margin to the other two class groups
- Reject or rewrite prompts whose within-group coherence is low.

3. Intensity ladder for class prompts

- Include prompt variants with weak vs strong sentiment:
  - mild positive, strongly positive
  - mild negative, strongly negative
- This may make the class prototypes better cover real tweet variation.

### Proposed measures for bias anchors

1. Valence-neutrality filter

- Use NRC VAD or Warriner-style valence norms to filter candidate bias anchors.
- Keep topic/style anchors only if their absolute valence is near neutral.
- This is especially important for emoticons, laughter, and exclamation-heavy forms.

2. Subjectivity filter

- Use MPQA-style strong/weak subjectivity or contextual polarity cues.
- Reject candidate bias anchors that are strongly subjective unless they are explicitly being used for an ablation.

3. Entropy + neutrality scoring

- For each candidate anchor phrase:
  - sentiment entropy across labels should be high
  - average affect / valence intensity should be low
- This is a better criterion than frequency alone.

4. Minimal-pair bias anchors

- Prefer pairs that differ in one nuisance attribute only:
  - URL vs no URL
  - question mark vs period
  - very short vs longer detailed
- Minimal pairs make the perturbation target cleaner.

5. Topic-first, style-second anchor scheduling

- First train with topic anchors, then optionally add style anchors.
- This may reduce the chance of stripping away genuine sentiment information early.

### Practical recommendation

For the next prompt revision, the safest “known quantities” setup is:

- class prompts:
  - lexicon-backed sentiment semantics
- primary bias anchors:
  - topic/entity anchors mined from data, filtered for sentiment neutrality
- secondary bias anchors:
  - URL / question / short-text structural cues
- ablation-only or low-priority anchors:
  - emoticons
  - laughter
  - exclamation-heavy cues

## Clarification: Where Supervision Enters The Current Pipeline

The current method is not purely label-free end-to-end.

What is label-free:

- the Phase 2 representation loss in `text_iccv`
- the PGD perturbation objective itself
- prompt-based prototype construction

Where labels currently enter:

1. Phase 2 checkpoint selection

- `cbdc/refine.py::_prepare_selector_data()` uses train and val labels
- `cbdc/refine.py::_selector_val_f1()` uses those labels to score checkpoints
- so the representation-training objective is label-free, but model selection is label-aware

2. Phase 3 probe evaluation

- `pipeline/classify.py` trains a linear probe using train labels and selects it using val labels

3. Final metric reporting

- both probe and prototype evaluation compare predictions to gold labels on val/test

### Clean summary

The current system is best described as:

- label-free prompt/perturbation training objective
- label-aware checkpoint selection in Phase 2
- either:
  - supervised downstream probe evaluation
  - or label-free prototype prediction with labeled measurement

So if a strict paper claim is needed, avoid saying “fully unsupervised end-to-end” in the current form.

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

## 2026-04-04 Follow-Up: Prompt Ratification And Supervision Audit

No compaction is needed right now, so prompt rationale is ratified here directly instead of being left as a TODO.

### Ratified now for `cbdc/prompts.py`

Current code locations:

- class prompts:
  - `cbdc/prompts.py::make_cls_text_groups()`
- CBDC style bias pairs:
  - `cbdc/prompts.py::CBDC_STYLE_BIAS_PAIRS`
- mined topic/style banks:
  - `cbdc/prompts.py::_CONTENT_TOPICS`
  - `cbdc/prompts.py::_STYLE_TOPICS`

Current ratification status:

- ratified as class prompts:
  - `negative / neutral / positive` sentiment prompts in `make_cls_text_groups()`
- ratified as primary bias-anchor families:
  - mined topic/entity anchors that are sentiment-neutral on the dataset
  - question-mark / repeated-question-mark style anchors
  - short-vs-long structural anchors
- ratified as useful but lower-confidence bias-anchor families:
  - URL / link style anchors
  - all-caps
  - elongated words
  - ellipsis / trailing-off style
- useful but risky and best treated as secondary or ablation-only:
  - emoticons
  - internet laughter
  - exclamation-heavy style

Reasoning:

- The dataset analysis strongly supports `class = sentiment`.
- The dataset analysis also supports several non-semantic surface correlations:
  - question marks
  - multiple question marks
  - URL/link
  - very short texts
- The analysis and prior tweet-sentiment literature both support social-media style cues such as:
  - all-caps
  - elongated words
  - punctuation sequences
  - emoticons
- However, emoticons / laughter / exclamation are partly sentiment-bearing, not just nuisance-bearing, so they are not the safest first-choice debias anchors.

### Practical ratification for the next `prompts.py` revision

If we revise the prompt bank, the safest concrete order is:

1. Keep `make_cls_text_groups()` as the class bank.
2. Keep mined topic/entity anchors as the primary `debias_vl` nuisance bank.
3. Keep structural pairs that are most defensible as nuisance signals:
   - question mark vs period
   - repeated question marks vs standard punctuation
   - URL/link vs no URL
   - very short vs longer detailed
4. Consider promoting these into the CBDC pair bank because they are already supported by tweet-sentiment literature:
   - all-caps vs normal case
   - elongated words vs ordinary spelling
   - ellipsis / trailing-off vs plain period
5. Demote these to secondary-bank or ablation-only status:
   - emoticon vs no emoticon
   - laughter vs no laughter
   - exclamation-heavy vs ordinary punctuation
6. Continue excluding direct polarity lexemes from bias anchors:
   - `love`, `happy`, `good`, `sad`, `hate`, `bad`, etc.

### Research-backed measures for better anchors and prompts

The most useful external guidance is not “find more sentiment words”, but “separate class-bearing affect from nuisance-bearing structure.”

High-confidence source-backed directions:

- NRC VAD gives continuous valence, arousal, and dominance scores and is explicitly intended for sentiment/emotion feature construction:
  - https://saifmohammad.com/WebPages/nrc-vad.html
- NRC Emotion Intensity gives degree information, which is useful for prompt intensity ladders:
  - https://www.saifmohammad.com/WebPages/AffectIntensity.htm
- Wilson, Wiebe, and Hoffmann (2009) is still a strong reference for contextual polarity and subjectivity:
  - https://aclanthology.org/J09-3003/
- SentiWordNet is still a defensible lexical polarity resource:
  - https://aclanthology.org/L06-1225/
- NRC-Canada’s tweet-sentiment system documents strong Twitter-native surface cues:
  - official summary page: https://saifmohammad.com/WebPages/NRC-Canada-Sentiment.htm
  - SemEval 2013 paper: https://aclanthology.org/S13-2053/
  - SemEval 2014 update: https://aclanthology.org/S14-2077/

Concrete measures suggested by those sources:

1. Valence-neutrality filter for bias anchors

- Score candidate topic/style anchors with NRC VAD.
- Keep anchors only if their average valence is near neutral.
- This is especially important before using:
  - emoticons
  - laughter
  - exclamation-heavy forms

2. Subjectivity filter for bias anchors

- Use MPQA/contextual-polarity style resources to reject highly subjective anchors unless they are explicitly part of an ablation.
- This helps keep the bias bank from accidentally swallowing class signal.

3. Intensity ladder for class prompts

- Build class prompt paraphrases that vary by strength:
  - mildly negative / strongly negative
  - mildly positive / strongly positive
- Use NRC VAD / Emotion Intensity to seed those paraphrases rather than hand-picking them arbitrarily.

4. Minimal-pair structural anchors

- Keep perturbation banks focused on single-attribute contrasts:
  - URL vs no URL
  - question mark vs period
  - repeated punctuation vs normal punctuation
  - all-caps vs ordinary casing
  - elongated spelling vs ordinary spelling
- This keeps the perturbation target cleaner than broad “style” descriptions.

5. Topic-first, style-second scheduling

- First train / perturb with topic-neutral anchors.
- Add structural anchors only after topic anchors are stable.
- This is the safest way to reduce the risk of removing genuine sentiment information too early.

### My current best recommendation

If the goal is to make the perturbation bank more defensible using known computational-linguistics quantities:

- class prompts:
  - keep the current sentiment bank, but expand it with lexicon-backed mild/strong variants
- primary bias anchors:
  - mined topic/entity anchors filtered by high label entropy and near-neutral valence
- secondary bias anchors:
  - question mark
  - repeated question marks
  - URL/link
  - very short text
  - all-caps
  - elongated words
  - ellipsis
- ablation-only anchors:
  - emoticons
  - laughter
  - exclamation-heavy forms
- do not use as bias anchors:
  - direct sentiment words
  - negators like `not` on their own
  - polarity shifters that obviously alter sentiment semantics

### Exact supervision entry points in the current code

The current pipeline is not “labels only in the final metric.” Labels enter in three earlier places.

1. Prompt/topic mining for `debias_vl` is label-aware

- In `cbdc/prompts.py`, `_label_entropy()` explicitly scores class balance.
- `mine_entity_topics_from_records()` counts labels per entity and keeps high-entropy entities.
- `mine_phrase_topics_from_records()` counts labels per phrase and keeps high-entropy phrases.
- `mine_curated_topic_rows_from_records()` also scores curated topic/style candidates against label distributions.

So:

- the mined `debias_vl` topic bank is not label-free
- it is selected using sentiment-label balance heuristics

2. Phase 2 checkpoint selection is label-aware

- In `cbdc/refine.py`, `_prepare_selector_data()` loads train and val labels.
- In `cbdc/refine.py`, `_centroid_val_f1()` builds class centroids from labeled train examples and scores on labeled val examples.
- In `cbdc/refine.py`, `_selector_val_f1()` uses that labeled F1 to choose the best CBDC checkpoint.

So:

- the Phase 2 loss itself is prompt-based and label-free
- but checkpoint selection inside Phase 2 is label-aware

3. Phase 3 probe evaluation is supervised

- In `pipeline/classify.py::train_linear_probe()`, `y_train` is used in cross-entropy optimization.
- `y_val` is also used to early-select the best probe checkpoint.

4. Prototype evaluation is label-free at prediction time, but labeled at measurement time

- `pipeline/prototype_classify.py` does not fit on labels.
- It compares embeddings to class prototypes and predicts by similarity.
- But val/test labels are still used to compute accuracy/F1.

### Clean framing to preserve

The most accurate current description is:

- label-aware topic/anchor mining for `debias_vl`
- label-free prompt/perturbation objective once the prompt banks are fixed
- label-aware checkpoint selection during Phase 2
- either:
  - supervised probe evaluation
  - or label-free prototype prediction with labeled evaluation

If a stricter “unsupervised” claim is needed later, the main remaining target is to replace the Phase 2 label-aware selector with a label-free selector.

### D2-specific clarification

For `D2 (CBDC)` under prototype evaluation:

- yes, this is the purest CBDC branch in the current codebase
- it uses the fixed prompt-only CBDC bank from `get_cbdc_prompt_bank()`
- it does not use the label-aware `debias_vl` topic miner
- it does not use DebiasVL-discovered anchors

But it is still not fully label-free end-to-end, because:

- the train/val/test split is stratified by sentiment labels
- Phase 2 checkpoint selection uses labeled train/val data via `selector_f1`
- prototype evaluation reports val/test accuracy and F1 against gold labels

So the most accurate phrasing is:

- `D2` uses a label-free CBDC training objective with fixed prompts
- but the full `D2` pipeline still includes label-aware model selection and labeled evaluation

### Plain-language comparison with `transformer-classifier-experiments`

The easiest way to explain the difference is:

- `transformer-classifier-experiments` is a normal labeled classifier branch
- the current CBDC pipeline is mostly a representation-shaping branch, with optional labeled evaluation on top

What the old branch does:

- It shows the model a real tweet and its true label:
  - negative
  - neutral
  - positive
- Then it updates the model to get that label right.
- In other words, it is directly trained to be good at tweet classification.

What the current CBDC pipeline does:

- It first builds text embeddings for the dataset.
- Then for `D2`, it trains using fixed prompt sentences and style pairs, not tweet-label pairs.
- The goal there is not “predict this tweet’s label correctly.”
- The goal is “reshape the embedding space so sentiment concepts and bias/style directions are separated better.”

What happens after that:

- if Phase 3 is `probe`, a small labeled classifier is trained on top of those embeddings
- if Phase 3 is `prototype`, no classifier is trained; the system just checks which class prototype a tweet is closest to

So in very plain terms:

- old branch:
  - teacher says “this tweet is positive”
  - model learns to say “positive”
- current `D2 + prototype`:
  - model is trained on prompt geometry, not tweet answers
  - afterwards we ask “is this tweet closer to the positive, neutral, or negative prototype?”

Important caveat:

- the current branch still uses labels for splitting the data, for selecting the best CBDC checkpoint, and for computing final scores
- so it is not fully label-free end to end
- but it is still very different from the old branch because it is not trained with the direct objective “make the tweet classifier predict the gold label”

### Safe wording for the paper / PI

What can be claimed safely:

- `D2 + prototype` is the closest thing in the current codebase to unsupervised training
- more precisely, it is:
  - a label-free prompt-based training objective
  - with labeled checkpoint selection
  - and labeled evaluation

What should not be claimed:

- not “fully unsupervised end-to-end”
- not “no labels are used”

Best safe phrasing:

- “label-free representation shaping with labeled model selection and evaluation”
- “prompt-based unsupervised objective, evaluated on labeled sentiment data”
- “weakly supervised overall pipeline; label-free CBDC objective”

### How strong the current prototype results are

From the RoBERTa prototype run:

- `B1 (raw)`:
  - test acc `0.4110`
  - test macro-F1 `0.3535`
- `D1 (debias_vl)`:
  - test acc `0.4541`
  - test macro-F1 `0.4418`
- `D2 (CBDC)`:
  - test acc `0.4442`
  - test macro-F1 `0.4451`

Meaningful deltas over raw:

- `D1`:
  - `+0.0432` accuracy
  - `+0.0883` macro-F1
- `D2`:
  - `+0.0332` accuracy
  - `+0.0916` macro-F1

Interpretation:

- these are meaningful effect sizes for a proof-of-concept
- especially because they appear in the prototype setting, which is the evaluation most aligned with CBDC’s prompt/prototype story
- however, they are not yet statistical-significance claims

What is still needed before using the word “significant” in the strict research sense:

- multiple random seeds
- mean and standard deviation
- ideally a paired test or bootstrap confidence interval on the test predictions

So the safest conclusion is:

- the current results are promising and materially better than the raw prototype baseline
- but they are not yet enough to claim statistical significance from a single run

## Original CBDC Paper: What “Unsupervised” Means

After reading `CBDC__CVPR_2026_final.pdf` and cross-checking the upstream code:

- the paper’s “without label supervision” claim refers to the debiasing method itself
- specifically, the text/image debiasing objective does not use class labels or bias labels as training targets
- instead, CBDC constructs bias directions from prompt geometry and PGD-perturbed representations

What the paper explicitly claims:

- it contrasts prior methods that use class or bias labels during training with methods that operate `w/o label`
- it describes CBDC as not requiring additional bias annotations
- worst-group accuracy, average accuracy, and gap are evaluation metrics, so they necessarily require labeled benchmark datasets

Important implementation nuance from the upstream reference code:

- the reference `direct_port` code does use labeled validation data for model selection
- in `base.py`, validation uses:
  - class labels (`target`)
  - spurious-group labels (`spurious`)
- the best checkpoint is chosen by validation robust / worst-group style performance

So the fairest reading is:

- original CBDC is unsupervised with respect to the debiasing objective
- but the benchmark experiments still rely on labeled validation/test sets to measure fairness metrics
- and the reference implementation also uses labeled validation metrics for checkpoint selection

This means the original paper is not best interpreted as:

- “no labels are touched anywhere”

It is better interpreted as:

- “the debiasing/training objective does not require label supervision”
- while labeled benchmark data is still used for evaluation, and in code also for choosing the best epoch

## 2026-04-04 Implementation Update: Fixed Test Split And D2.5

### Dataset loader behavior

`dataset.py` now supports a fixed local `test.csv` split.

Current behavior:

- if `project/data/test.csv` does not exist:
  - `train.csv` is split into train / val / test as before
- if `project/data/test.csv` exists:
  - `train.csv` is split only into train / val
  - `test.csv` is used as the fixed test set

This is now the correct setup for the sentiment benchmark story the user wanted:

- training data in `train.csv`
- held-out benchmark data in `test.csv`

### D2.5 experiment

Added an opt-in condition:

- `D2.5 (CBDC no-label-select)`

Meaning:

- same fixed-prompt CBDC objective as `D2`
- same prototype / probe evaluation options as other conditions
- but Phase 2 checkpoint selection is not based on labeled validation F1

Current selector choices:

- `D2`:
  - checkpoint selected by labeled validation centroid macro-F1
- `D2.5`:
  - checkpoint selected by lowest prompt training loss
  - no label-based selector is used during Phase 2

This makes `D2.5` the cleanest current experiment for the desired story:

- fixed prompts
- no `debias_vl` label-mined anchors
- no label-based checkpoint selector
- labels only for split construction and final evaluation

### How to run

Local:

```bash
python run_all.py --model roberta --classifier prototype --include_d25
```

SLURM:

```bash
sbatch --export=ALL,CLASSIFIER=prototype,INCLUDE_D25=1 submit_new.slurm
```

### Important interpretation

`D2.5` is still not literally “no labels touched anywhere” because:

- the dataset loader still expects sentiment labels in the CSV
- train/val splitting is still stratified by sentiment labels
- final reporting still uses labeled val/test data

But compared with `D2`, it removes one major label-aware step:

- no labeled checkpoint selection inside Phase 2

So if a cleaner claim is needed, the safest phrasing is:

- `D2.5`: label-free CBDC objective with label-free checkpoint selection, evaluated on labeled sentiment data

### Audit results

Status:

- audited and smoke-tested after implementation
- core code paths now work as intended for:
  - fixed local `test.csv`
  - `D2.5` materialization
  - prototype evaluation including `D2.5`
  - final evaluation report including `D2.5`

What was actually tested:

1. Syntax / import checks

```bash
python -m compileall dataset.py cbdc/refine.py pipeline/artifacts.py pipeline/evaluate.py run_all.py run_all_debug.py
python run_all.py --help
python run_all_debug.py --help
bash -n submit_new.slurm
```

2. Fixed `test.csv` loader path

- Used a temporary directory with synthetic `train.csv` and `test.csv`
- Monkeypatched `dataset.LOCAL_DATA_DIR` in-process
- Confirmed behavior:
  - `train.csv` -> split into train / val only
  - `test.csv` -> used directly as the held-out test set

3. D2.5 smoke test

- Built a tiny temporary cache with synthetic train / val / test splits
- Ran:

```bash
CACHE_DIR=<tmp_cache> MODEL_NAME=bert-base-uncased INCLUDE_D25=1 \
python cbdc/refine.py --n_epochs 1 --eval_every 1 --use_static_topics \
  --selector_train_per_class 1 --selector_batch_size 2 \
  --mine_max_topics 4 --pole_phrases_per_side 2 --n_bias_dirs 2
```

- Confirmed:
  - `B1`, `D1`, `D2`, `D2.5`, and `D3` all materialize
  - `D2` logs `Checkpoint selector: labeled val centroid macro-F1`
  - `D2.5` logs `Checkpoint selector: prompt loss only (label-free)`
  - all condition splits and prototypes are written successfully

4. Prototype evaluation + report with `D2.5`

```bash
CACHE_DIR=<tmp_cache> INCLUDE_D25=1 python pipeline/prototype_classify.py

CACHE_DIR=<tmp_cache> INCLUDE_D25=1 \
RESULTS_FILE=results_prototype.pt \
REPORT_FILE=eval_report_prototype_smoke.txt \
EVAL_TITLE='Smoke Prototype Report' \
RESULTS_SECTION_TITLE='Prototype Results' \
python pipeline/evaluate.py
```

- Confirmed:
  - `prototype_classify.py` includes `D2.5`
  - `evaluate.py` includes `D2.5`
  - comparative analysis prints `D2.5 vs B1` and `D2.5 vs D2`

### Issue found during audit

One real issue was found and fixed:

- when `source='local'` and a local CSV existed but loading/splitting failed, `dataset.py` raised a misleading `FileNotFoundError`
- this is now fixed so the actual local load failure is surfaced as a `RuntimeError`

### Practical note

On the current local workspace, there is no real `project/data/test.csv`, so the live repo still falls back to the old train/val/test split here.

To activate the fixed test behavior in an actual run:

- place the original benchmark file at:
  - `project/data/test.csv`
- then run the pipeline normally

### Bottom-line audit conclusion

- yes, the code changes themselves work as intended
- `D2.5` is operational
- fixed `test.csv` support is operational
- the remaining caveat is experimental, not code-level:
  - the fixed-test behavior only activates when a real `test.csv` is present

- Report two settings separately:
  - prompt/prototype zero-shot classification
  - supervised linear-probe evaluation
- This avoids mixing the "unsupervised training" story with the "supervised downstream probe" story.

### D2 vs D2.5 artifact isolation

- `D2` and `D2.5` do not share condition-specific `.pt` artifacts.
- They are written into different condition directories under the same model cache:
  - `conditions/d2_cbdc/`
  - `conditions/d25_cbdc_no_label_select/`
- So these files are isolated per condition:
  - `class_prompt_prototypes.pt`
  - `encoder_cbdc.pt`
  - `training_meta.json`
  - `z_tweet_train.pt`
  - `z_tweet_val.pt`
  - `z_tweet_test.pt`
- They do intentionally share the raw Phase 1 cache at the model root:
  - `z_tweet_train.pt`
  - `z_tweet_val.pt`
  - `z_tweet_test.pt`
- That shared root cache is the common input to all conditions and is not cross-contamination by itself.
- They also use the same fixed CBDC prompt bank definition, but each condition re-encodes and saves its own prompt prototypes separately.
- `cbdc/refine.py` loads a fresh encoder for `D2` and another fresh encoder for `D2.5`, so there is no parameter carry-over between the two branches during materialization.

## 2026-04-04 Fixed-Test Prototype Run With D2.5

Run setting:

- model: `roberta`
- classifier: `prototype`
- `INCLUDE_D25=1`
- fixed held-out `project/data/test.csv` was present and used

Observed split:

- train: `23358`
- val: `4122`
- test: `3534`

Prototype summary on the fixed test split:

- `B1 (raw)`:
  - test acc `0.4318`
  - test macro-F1 `0.3747`
- `D1 (debias_vl)`:
  - test acc `0.4709`
  - test macro-F1 `0.4577`
- `D2 (CBDC)`:
  - test acc `0.4709`
  - test macro-F1 `0.4715`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.4601`
  - test macro-F1 `0.4613`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.4349`
  - test macro-F1 `0.3750`

Key interpretation:

- `D2` and `D1` tie on test accuracy, but `D2` has the best macro-F1 in the run.
- `D2.5` is worse than `D2`, which suggests the labeled validation selector is still helping choose a stronger checkpoint.
- But `D2.5` still clearly beats `B1`, which is the important proof point for the label-free-selector story.

Useful deltas:

- `D2` vs `B1`:
  - accuracy `+0.0390`
  - macro-F1 `+0.0968`
- `D2.5` vs `B1`:
  - accuracy `+0.0283`
  - macro-F1 `+0.0866`
- `D2.5` vs `D2`:
  - accuracy `-0.0108`
  - macro-F1 `-0.0102`

Meaning:

- The CBDC objective itself is doing useful work, because even without label-based checkpoint selection, `D2.5` remains materially better than the raw prototype baseline.
- The labeled selector in `D2` adds another small gain on top of that.
- So the cleanest current claim is:
  - `D2.5` supports a label-free-training / labeled-evaluation story
  - `D2` is the stronger-performing but slightly less clean version because checkpoint selection uses labels

Important caution:

- This fixed-test run should be compared primarily within itself.
- It should not be over-interpreted against older runs that used the previous random train/val/test split, because the evaluation set changed.

## 2026-04-04 Fixed-Test Prototype Run With D2.5 (Run 2)

Run setting:

- model: `roberta`
- classifier: `prototype`
- `INCLUDE_D25=1`
- fixed held-out `project/data/test.csv` was present and used

Observed split:

- train: `23358`
- val: `4122`
- test: `3534`

Prototype summary on the fixed test split:

- `B1 (raw)`:
  - test acc `0.4318`
  - test macro-F1 `0.3747`
- `D1 (debias_vl)`:
  - test acc `0.4709`
  - test macro-F1 `0.4577`
- `D2 (CBDC)`:
  - test acc `0.4709`
  - test macro-F1 `0.4719`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.4513`
  - test macro-F1 `0.4503`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.4363`
  - test macro-F1 `0.3764`

Useful deltas:

- `D2` vs `B1`:
  - accuracy `+0.0390`
  - macro-F1 `+0.0972`
- `D2.5` vs `B1`:
  - accuracy `+0.0195`
  - macro-F1 `+0.0756`
- `D2.5` vs `D2`:
  - accuracy `-0.0195`
  - macro-F1 `-0.0216`

Key interpretation:

- `D2` again beats `B1` by a large margin.
- `D2.5` again beats `B1`.
- `D2` again beats `D2.5`.
- This repeats the same ordering as the previous fixed-test run.

## Repeatability note across the two fixed-test runs so far

Common pattern:

- `B1` remains the weakest meaningful baseline.
- `D2` is consistently one of the top methods and has the strongest macro-F1.
- `D2.5` is consistently better than `B1`, but consistently below `D2`.
- `D3` remains weak.

What this supports:

- The CBDC objective itself appears to help even when checkpoint selection is label-free, because `D2.5 > B1` in both fixed-test runs.
- The labeled validation selector appears to add additional value, because `D2 > D2.5` in both fixed-test runs.

Safe wording if this trend continues over more runs:

- On a fixed held-out test set, the label-free-selector variant `D2.5` consistently improves over the raw prototype baseline.
- The label-selected `D2` variant consistently provides an additional gain over `D2.5`.

## 2026-04-04 Large-model support and stability guardrails

Motivation:

- The original CBDC paper relies on a strong backbone with rich representational structure.
- For the NLP port, larger encoders are reasonable to test, but only if the custom Phase 2 latent-tail path is actually architecture-compatible.

What changed:

- Added safe registry shortcuts for larger encoder models:
  - `roberta-large` -> `FacebookAI/roberta-large`
  - `xlmr-large` -> `FacebookAI/xlm-roberta-large`
- Forced backbone loading in `float32` for stability.
- Added gradient clipping for the trainable tail during Phase 2:
  - `max_grad_norm=1.0` by default
- Added a non-finite-loss guard during CBDC training.

Important architecture note:

- Phase 1 embedding extraction can work with many Hugging Face encoder models.
- Phase 2 CBDC latent-tail training is now explicitly restricted to the safe BERT-like families that match the current tail-forwarding assumptions:
  - `bert`
  - `roberta`
  - `xlm-roberta`
  - `distilbert`
- This avoids silent breakage for architectures whose layer forward signatures differ in ways that matter for the custom tail pass.

Why some large models are intentionally not yet enabled for Phase 2:

- `microsoft/deberta-v3-large` uses DeBERTa-v2/V3 layer internals with extra relative-position arguments in the layer forward path.
- `answerdotai/ModernBERT-large` uses a different encoder-layer interface and longer-context setup.
- Those models may be excellent candidates later, but they need architecture-specific Phase 2 support rather than being dropped into the current path naively.

Smoke check:

- `roberta-base` Phase 2 path still works after the stability changes:
  - model loads in `float32`
  - intermediate hidden extraction works
  - delta/tail path works

Recommended next large-model commands:

- `sbatch --export=ALL,RUN_MODEL=roberta-large,CLASSIFIER=prototype,INCLUDE_D25=1 submit_new.slurm`
- `sbatch --export=ALL,RUN_MODEL=xlmr-large,CLASSIFIER=prototype,INCLUDE_D25=1 submit_new.slurm`

Why not Qwen / Llama in the current pipeline:

- Qwen2.5 and Llama 3.1 are causal language models, not encoder-style MLM backbones.
- The current Phase 2 CBDC port assumes:
  - bidirectional encoder representations
  - CLS-style sentence pooling at token position 0
  - a BERT-like latent-tail path where delta is injected before the final encoder layer
- Decoder-only models violate those assumptions:
  - no natural CLS token
  - causal masking instead of bidirectional encoding
  - different layer/attention semantics for the custom tail pass
- So using Qwen/Llama in the current code would not be a simple “bigger model” swap; it would require a separate decoder-style Phase 2 implementation.

Practical conclusion:

- If the goal is a stronger backbone while staying faithful to the current CBDC-NLP mechanics, large encoder models are the safest next step.
- If the goal is to test Llama/Qwen specifically, that should be treated as a new method branch, not a drop-in replacement.

Safe large-model shortlist for the current code:

- `FacebookAI/roberta-large`
  - English
  - about `0.4B` params on the HF card
  - best first choice for a larger CBDC-style run
- `google-bert/bert-large-cased`
  - English
  - `24` layers, `1024` hidden, `336M` params on the HF card
  - safe to run by passing the full HF model ID
- `FacebookAI/xlm-roberta-large`
  - multilingual
  - about `0.6B` params on the HF card
  - largest safe encoder currently supported in the pipeline

Recommended order:

1. `roberta-large`
2. `bert-large-cased`
3. `xlm-roberta-large`

Added uncased BERT shortcuts:

- `bert-uncased` -> `google-bert/bert-base-uncased`
- `bert-base-uncased` -> `google-bert/bert-base-uncased`
- `bert-large-uncased` -> `google-bert/bert-large-uncased`

Important note:

- plain `bert` still maps to `bert-base-cased`
- the new uncased aliases are additive and do not change existing experiment behavior

## 2026-04-04 Decoder-only alternative path (Qwen / Llama-style)

Motivation:

- The user wants stronger backbones beyond encoder-style BERT/RoBERTa models.
- To stay as semantically close as possible to CBDC, the decoder-only variant should still:
  - use a frozen backbone
  - extract an intermediate latent state
  - inject a perturbation before the final transformer block(s)
  - train only the tail path used in Phase 2

What was implemented:

- Added a decoder-family Phase 2 path for:
  - `qwen2`
  - `llama`
- Main encoder changes:
  - decoder models use last non-pad token pooling instead of CLS pooling
  - delta is injected at the last non-pad token, not token position 0
  - tail forwarding uses causal masks and rotary embeddings consistent with the model's own forward pass
  - decoder families apply the final model norm after the tail layers before pooling

Model shortcuts added:

- `qwen25-0.5b` -> `Qwen/Qwen2.5-0.5B`
- `qwen25-1.5b` -> `Qwen/Qwen2.5-1.5B`
- `qwen25-3b` -> `Qwen/Qwen2.5-3B`
- `llama32-1b` -> `meta-llama/Llama-3.2-1B`
- `llama32-3b` -> `meta-llama/Llama-3.2-3B`

Important safety fix:

- `data/embed.py` no longer silently falls back to DistilBERT when a requested model fails to load.
- This is especially important for gated decoder models like Llama, where a silent fallback would corrupt the experiment.

Local verification status:

- Existing encoder-family path still works:
  - `roberta-base` smoke test passed
- Decoder-family path verified locally for Qwen:
  - `Qwen/Qwen2.5-0.5B` smoke test passed
  - tokenization, encoding, intermediate-state extraction, and delta-tail path all completed successfully
- Llama path is wired, but not locally validated in this environment because Meta checkpoints are gated without HF authentication

Current best practical decoder-family recommendation:

- `qwen25-3b`
  - closest to the requested “~2.5B-level” size
  - open and not gated like Meta Llama

Example command:

- `sbatch --export=ALL,RUN_MODEL=qwen25-3b,CLASSIFIER=prototype,INCLUDE_D25=1 submit_new.slurm`

Methodological note:

- This decoder-family path is not identical to the original encoder-style NLP port.
- But it is semantically consistent with the paper at the level that matters:
  - perturb latent representation
  - derive bias directions from prompt responses
  - debias via the model's final textual representation rather than direct supervised label fitting

## 2026-04-04 CLIP-to-NLP Path Similarity Audit

Files audited:

- Original CBDC / CLIP side:
  - `direct_port/base.py`
  - `direct_port/train_bafa.py`
  - `direct_port/can/attack/simple_pgd.py`
- Local NLP side:
  - `encoder.py`
  - `cbdc/refine.py`
  - `losses.py`
  - `cbdc/prompts.py`

Bottom-line verdict:

- The current NLP CBDC port is not a superficial analogy.
- It mirrors the original CLIP text-stage structure quite closely at the level of:
  - frozen-body + trainable-tail decomposition
  - latent-space PGD over prompt representations
  - bipolar adversarial perturbation discovery
  - match-loss and cross-knowledge training
- But it is still not a line-by-line port.
- The biggest approximation is:
  - original CLIP PGD perturbs the full intermediate sequence tensor
  - local NLP PGD perturbs a single token-position latent delta that is injected into the hidden sequence before the final layer

Important clarification about the CLIP reference:

- When describing similarity, it is safer to anchor the analogy to the RN50-style target-tail setup used in the original CBDC code, not to a generic "projection-layer mirror".
- In the released code, the important common pattern is:
  - frozen encoder body
  - expose an intermediate latent
  - perturb that latent
  - run the last target block(s) to get the final normalized embedding
- For CLIP image models:
  - ViT variants have a direct `proj`
  - RN50 variants instead go through `attnpool` and `c_proj`
- So the local NLP port should be described as mirroring the target-tail latent path, not as specifically mirroring a ViT-style projection head.

### Direct path correspondence

Original CLIP / CBDC:

- `Collaboration.text_iccv(...)` in `base.py`
  - main text-stage training loop
- `SetTarget.get_feature(...)` in `base.py`
  - extract intermediate text hidden states before the last transformer block
- `SetTarget.forward(...)` in `base.py`
  - run the final text block(s), apply final norm / projection, then pool to the text embedding
- `perturb_bafa_txt_multi_ablation_lb_ls(...)` in `simple_pgd.py`
  - bipolar latent PGD with:
    - sign updates
    - random restarts
    - `att_loss`
    - optional `keep_loss`

Local NLP / CBDC:

- `text_iccv(...)` in `cbdc/refine.py`
  - main text-stage training loop
- `get_intermediate_features(...)` in `encoder.py`
  - extract intermediate hidden states before the final transformer layer
- `encode_with_delta_from_hidden(...)` in `encoder.py`
  - run the final transformer block(s), then pool to the sentence embedding
- `_pgd_bipolar(...)` in `cbdc/refine.py`
  - bipolar latent PGD with:
    - sign updates
    - random restarts
    - `L_B`
    - `L_s`

### Where the mirroring is strong

1. Frozen backbone + last-layer tail training

- Original CLIP code:
  - `txt_model = SetTarget(..., 1, 'txt', cfg.txt_learn_mode)` in `base.py`
  - with `learn_mode='linear'`, only the last text transformer block is trainable
- Local NLP code:
  - `layer_tail = encoder._get_transformer_layers()[-1]` in `cbdc/refine.py`
  - only that last layer is unfrozen and optimized

Conclusion:

- This is highly analogous.
- Both methods treat the backbone as frozen and adapt only the terminal text block.

2. Intermediate-latent extraction before the tail

- Original CLIP text side:
  - `SetTarget.get_feature(...)` runs all but the last text block and returns the latent sequence `z`
- Local NLP side:
  - `get_intermediate_features(...)` returns the hidden sequence before the final transformer layer

Conclusion:

- This is one of the cleanest mirrors in the whole port.

3. Latent PGD with bipolar perturbation branches

- Original:
  - `perturb_bafa_txt_multi_ablation_lb_ls(...)` creates two adversarial sets:
    - one pushed toward pole A
    - one pushed toward pole B
  - uses:
    - sign-gradient PGD
    - restart noise
    - bounded perturbations
- Local:
  - `_pgd_bipolar(...)` explicitly produces:
    - `z_adv_pos`
    - `z_adv_neg`
  - also uses:
    - sign-gradient PGD
    - restart noise
    - bounded perturbations

Conclusion:

- This is strongly mirrored.

4. Loss formulas

- Original `simple_pgd.py`:
  - `att_loss = cross_entropy(100 * logits, target)`
  - `keep_loss = 100 * ((adv_feat - ori) @ keep.T).pow(2).mean()`
  - `loss = att_loss * (1 - keep_weight) - keep_loss * keep_weight`
- Local `losses.py` + `_pgd_bipolar(...)`:
  - `L_B` is the same pairwise CLIP-style cross-entropy over bias-pole pairs
  - `L_s` explicitly matches the original keep-loss formula
  - `loss = L_B * (1 - keep_weight) - L_s * keep_weight`
- Original `base.py` text training:
  - `match_loss` from `S = adv_cb_set1 - adv_cb_set2`
  - `ck_loss = ((bias_a - bias_b) @ cls_em.T).pow(2).mean() * up_`
- Local `cbdc/refine.py` + `losses.py`:
  - `S = z_adv_pos - z_adv_neg`
  - `match_loss` uses the same class-alignment structure
  - `l_ck(...)` explicitly implements the original `ck_loss` formula

Conclusion:

- The mathematical structure is very close.
- This is the strongest evidence that the NLP port is genuinely modeled after the original CBDC text stage.

### Main differences from the original CLIP path

1. Full-sequence perturbation vs single-token injected delta

- Original CLIP:
  - PGD perturbs the entire latent sequence tensor `z`
  - `z_adv1` has the same shape as the whole intermediate text representation
- Local NLP:
  - PGD optimizes a `delta` of shape `(B, H)`
  - that delta is injected only at one pooled-token position:
    - CLS for encoder families
    - last non-pad token for decoder families

This is the single biggest approximation in the local port.

Interpretation:

- The local method is still latent-tail perturbation.
- But it is a lower-dimensional, pooled-token-centered perturbation, not a full hidden-sequence perturbation.

2. Pooling semantics differ by backbone family

- Original CLIP text encoder:
  - pools at the end-of-text position (`token_max`) before `text_projection`
- Local encoder-family models:
  - pool from CLS
- Local decoder-family models:
  - pool from the last non-pad token

Important nuance:

- The decoder-family Qwen/Llama path is actually closer to CLIP text pooling semantics than the BERT/RoBERTa CLS path.

3. VLM collaboration vs text-only port

- Original paper/code:
  - alternates between text-side and image-side training
  - uses image embeddings for evaluation and worst-group robustness
- Local NLP port:
  - only keeps the text-stage idea
  - there is no image-side collaboration stage
  - final evaluation is text-only

Conclusion:

- The local method is best described as a text-only adaptation of the CLIP text-stage CBDC logic, not a full multimodal reproduction.

4. Validation / model selection is only partially mirrored

- Original CLIP code:
  - `val_()` evaluates image-text zero-shot predictions
  - best checkpoint is chosen by robust / worst-group validation behavior
- Local NLP `D2`:
  - best checkpoint is chosen by labeled validation centroid macro-F1
- Local NLP `D2.5`:
  - best checkpoint is chosen by prompt loss only

Conclusion:

- `D2` is structurally similar in the sense of label-aware validation selection, but the metric is different.
- `D2.5` is cleaner for a label-free story, but less similar to the released CLIP code's selector behavior.

5. Precision / implementation details differ

- Original CLIP code makes heavy use of half precision (`.half()`)
- Local NLP path was intentionally stabilized toward `float32`
  - especially for larger encoder and decoder models

Conclusion:

- This is an engineering divergence for stability, not a conceptual divergence in the method.

### Safe summary wording

- The current NLP CBDC port is strongly inspired by and structurally close to the original CLIP text-stage CBDC code.
- It mirrors:
  - the frozen-body / trainable-tail split
  - intermediate-latent extraction
  - bipolar latent PGD
  - semantic-preservation loss
  - match-loss and cross-knowledge training
- The main deviation is that the local NLP port perturbs a pooled-token latent delta rather than the full intermediate sequence tensor.
- So the right claim is:
  - semantically faithful adaptation of the CLIP text-stage CBDC path
  - not an exact line-by-line reproduction
- If wording needs to be especially careful, prefer:
  - "RN50-style frozen-body / target-tail latent adaptation"
  - rather than
  - "same projection-layer path as CLIP"

### Perturbation-depth note

Current state:

- The local code already has the structural hook for deeper tail execution:
  - `encode_with_delta_from_hidden(..., start_layer=...)` in `encoder.py`
- But the current training loop always uses the default:
  - perturb before the last layer
  - unfreeze only the final transformer layer

So there are three different meanings of "increase perturbation depth":

1. More PGD iterations

- Safe and already supported by:
  - `n_pgd_steps`
- This does not change where the perturbation enters, only how hard PGD pushes it.

2. Deeper tail span

- Most natural next step.
- Change the port so that:
  - `get_intermediate_features(...)` stops earlier
  - `encode_with_delta_from_hidden(..., start_layer=...)` runs the last 2 or 3 layers
  - those same last 2 or 3 layers are unfrozen for optimization
- This would make the NLP path closer to a stronger target-tail adaptation without changing the rest of the method.

3. Full-sequence perturbation instead of single-token delta

- This would be the closest move toward the original CLIP mechanics.
- But it is the biggest change and the riskiest:
  - much more memory
  - much easier to destabilize training
  - much more likely to damage semantics unless `L_s` and step sizes are retuned

Practical recommendation:

- Increasing depth probably will not "break the method" if done moderately.
- The safest version is:
  - keep the current single-token delta
  - move the perturbation insertion point earlier
  - train the last 2 layers instead of only the last 1
- That is a reasonable D2-depth ablation.

Main risk:

- Because the local perturbation is a single injected vector, pushing it through too many layers can make it either:
  - wash out before the output
  - or over-distort the representation and hurt sentiment structure

So the recommended order is:

1. try more `n_pgd_steps` first
2. then try a `tail_layers=2` version
3. only later consider full-sequence perturbation

Safe interpretation:

- Moderate depth increase is method-consistent.
- Large depth increase is possible, but becomes a new variant rather than a small tweak.

## 2026-04-04 Additional Fixed-Test Runs

### RoBERTa fixed-test run (additional)

- `B1 (raw)`:
  - test acc `0.4318`
  - test macro-F1 `0.3747`
- `D1 (debias_vl)`:
  - test acc `0.4709`
  - test macro-F1 `0.4577`
- `D2 (CBDC)`:
  - test acc `0.4740`
  - test macro-F1 `0.4758`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.4544`
  - test macro-F1 `0.4411`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.4349`
  - test macro-F1 `0.3750`

Interpretation:

- This is another RoBERTa run where:
  - `D2 > B1`
  - `D2.5 > B1`
  - `D2 > D2.5`
- `D2` reaches the strongest RoBERTa macro-F1 observed so far on the fixed test split.

### BERT-base-cased fixed-test run

- `B1 (raw)`:
  - test acc `0.4287`
  - test macro-F1 `0.3785`
- `D1 (debias_vl)`:
  - test acc `0.4810`
  - test macro-F1 `0.4782`
- `D2 (CBDC)`:
  - test acc `0.4621`
  - test macro-F1 `0.4574`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.4581`
  - test macro-F1 `0.4074`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.4290`
  - test macro-F1 `0.3930`

Interpretation:

- On BERT, `D1` is the strongest method.
- `D2` and `D2.5` still improve over `B1`, but the advantage is smaller than on RoBERTa.
- `D2 > D2.5` still holds.

## Cross-model read so far

- RoBERTa appears to be the stronger backbone for the CBDC-style path:
  - `D2` is especially strong on RoBERTa.
- BERT appears to favor the DebiasVL-style path more:
  - `D1` is strongest there.
- Across both backbones, one robust pattern remains:
  - `D2 > B1`
  - `D2.5 > B1`
  - `D2 > D2.5`

Current safe conclusion:

- The label-free-selector CBDC variant (`D2.5`) continues to improve over the raw prototype baseline across backbones.
- The label-selected CBDC variant (`D2`) continues to provide an additional gain over `D2.5`.
- The relative ranking of `D1` vs `D2` is backbone-dependent.

## Proposed Next Tests

### Closest-to-CBDC-paper tests

1. Repeatability on a fixed held-out test set

- Run multiple seeds for:
  - `B1`
  - `D2`
  - `D2.5`
- Report mean and std for:
  - test accuracy
  - macro-F1
- This is the cleanest next step for making the current prototype story reportable.

2. Fairness/robustness-style subgroup evaluation

- The original paper emphasizes:
  - worst-group accuracy
  - average accuracy
  - performance gap
- NLP analogue:
  - define heuristic subgroups from bias cues on the held-out test set
  - examples:
    - URL present / absent
    - question-mark ending / not
    - repeated question marks / not
    - very short / not
    - topic-present / topic-absent
- Then report:
  - average accuracy
  - worst-group accuracy
  - gap = avg - worst-group

3. Utility-preservation evaluation

- The original paper explicitly checks whether debiasing preserves zero-shot utility on unrelated benchmarks.
- NLP analogue:
  - after learning D2 / D2.5 on the main tweet sentiment data, test prototype classification on an external sentiment dataset or domain-shifted dataset without retraining the method
  - the goal is to show that debiasing does not destroy general sentiment utility

4. PCA / embedding-organization visualization

- The original paper shows before/after embedding organization.
- NLP analogue:
  - PCA or UMAP of test embeddings
  - color by sentiment
  - marker shape by heuristic bias group
- Desired pattern:
  - after D2/D2.5, points cluster more by sentiment and less by nuisance group

5. Bias-direction induction sanity check

- The original paper qualitatively verifies that PGD-induced directions actually control meaningful bias attributes.
- NLP analogue:
  - for a handful of prompts or tweets, inspect nearest neighbors before and after moving along a learned bias direction
  - check whether the shift changes style/topic cues while preserving sentiment semantics

### Strong local ablations

1. D2 vs D2.5

- This is already one of the most useful local ablations:
  - same CBDC objective
  - label-based selector vs label-free selector

2. Semantic-preservation ablation

- Vary `keep_weight`
- Recommended settings:
  - `0.0`
  - `0.5`
  - default
- Goal:
  - test whether semantic preservation is what prevents class collapse

3. Sentiment-orthogonal PGD ablation

- Run with and without `sent_orthogonal_pgd`
- This is one of the clearest method contributions in the local NLP version.

4. PGD strength sweep

- Sweep:
  - `n_pgd_steps`
  - `num_samples`
  - `random_eps`
- Goal:
  - check whether stronger perturbation helps bias discovery or just damages sentiment structure

5. Bias-bank size / diversity sweep

- Sweep:
  - `n_bias_dirs`
  - `mine_max_topics`
  - `pole_phrases_per_side`
- Goal:
  - test whether more diverse bias directions actually help or whether noisy anchors start hurting performance

6. Prompt-bank ablation

- Compare:
  - current fixed CBDC bank
  - safer structural-only bank
  - topic-heavy bank
  - risky emoticon/laughter bank as ablation only
- Goal:
  - show which bias anchors help without eating sentiment signal

### Recommended priority order

1. Multi-seed fixed-test runs for `B1`, `D2`, `D2.5`
2. Worst-group / gap evaluation with heuristic NLP bias groups
3. `keep_weight` and `sent_orthogonal_pgd` ablations
4. External-dataset utility-preservation test
5. PCA / UMAP visualization

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

## 2026-04-04 BERTweet fixed-test prototype run

Model:

- `vinai/bertweet-base`
- classifier: `prototype`
- `INCLUDE_D25=1`

Results:

- `B1 (raw)`:
  - test acc `0.3305`
  - test macro-F1 `0.3000`
- `D1 (debias_vl)`:
  - test acc `0.3483`
  - test macro-F1 `0.3441`
- `D2 (CBDC)`:
  - test acc `0.3373`
  - test macro-F1 `0.3269`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.3430`
  - test macro-F1 `0.3349`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.3444`
  - test macro-F1 `0.3131`

Interpretation:

- Absolute prototype performance is much weaker than RoBERTa or BERT on this setup.
- `D1` is the best BERTweet method in this run.
- `D2.5` is slightly better than `D2` on BERTweet, which is the opposite of the stronger RoBERTa/BERT pattern.
- So the currently stable cross-backbone claim should stay modest:
  - all methods are somewhat backbone-dependent
  - `D2.5 > B1` still holds here
  - but `D2 > D2.5` is not universal across all backbones

Likely practical note:

- BERTweet may want prompt wording closer to tweets rather than generic `text`.
- If BERTweet is explored further, try the same run with:
  - `--text_unit tweet`
- It is also possible that BERTweet is simply worse for this prototype-prompt setup even if it is tweet-native.

Updated cross-model read:

- RoBERTa currently gives the strongest CBDC-style results.
- BERT gives strong DebiasVL-style results.
- BERTweet is weaker overall in the current prototype setup.

## 2026-04-04 BERT-uncased fixed-test prototype run

Model:

- `google-bert/bert-base-uncased`
- classifier: `prototype`
- `INCLUDE_D25=1`

Results:

- `B1 (raw)`:
  - test acc `0.3812`
  - test macro-F1 `0.3081`
- `D1 (debias_vl)`:
  - test acc `0.4440`
  - test macro-F1 `0.4087`
- `D2 (CBDC)`:
  - test acc `0.3902`
  - test macro-F1 `0.3289`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.3854`
  - test macro-F1 `0.3191`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.3721`
  - test macro-F1 `0.2877`

Interpretation:

- This is a weak result for the CBDC path on uncased BERT.
- `D1` is again the clear winner on the BERT family here.
- `D2` and `D2.5` only slightly improve over raw `B1`, and the margins are much smaller than with `bert-base-cased`.
- `D3` is actively worse than raw on this run.

Direct comparison against the earlier `bert-base-cased` fixed-test run:

- `bert-base-cased`:
  - `B1` acc `0.4287`, F1 `0.3785`
  - `D1` acc `0.4810`, F1 `0.4782`
  - `D2` acc `0.4621`, F1 `0.4574`
  - `D2.5` acc `0.4581`, F1 `0.4074`
- `bert-base-uncased`:
  - `B1` acc `0.3812`, F1 `0.3081`
  - `D1` acc `0.4440`, F1 `0.4087`
  - `D2` acc `0.3902`, F1 `0.3289`
  - `D2.5` acc `0.3854`, F1 `0.3191`

Takeaway:

- Lowercasing is not helping this prototype setup.
- The gap is large enough that uncased BERT should not be prioritized over cased BERT or RoBERTa for the current method.
- Plausible inference:
  - case information may still matter for the sentiment / prototype geometry in this dataset
  - or the cased BERT backbone is simply a better fit than the uncased one for this prompt bank
- Safe wording:
  - "BERT-uncased underperformed BERT-cased substantially in the current prototype CBDC setting."

## 2026-04-04 BERT-large-uncased fixed-test prototype run

Model:

- `google-bert/bert-large-uncased`
- classifier: `prototype`
- `INCLUDE_D25=1`

Results:

- `B1 (raw)`:
  - test acc `0.3656`
  - test macro-F1 `0.3654`
- `D1 (debias_vl)`:
  - test acc `0.4211`
  - test macro-F1 `0.4080`
- `D2 (CBDC)`:
  - test acc `0.3687`
  - test macro-F1 `0.3135`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.3995`
  - test macro-F1 `0.3620`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.3947`
  - test macro-F1 `0.3931`

Interpretation:

- This does not rescue the uncased BERT family for the current CBDC path.
- `D1` is still the strongest method.
- `D2` is especially weak here:
  - it almost collapses on neutral recall
  - macro-F1 is worse than raw `B1`
- `D2.5` is much healthier than `D2` on this model and clearly better than raw `B1` on accuracy, but still not close to the best cased-BERT or RoBERTa results.
- `D3` is better than on base uncased and is surprisingly competitive here, though still below `D1`.

Comparison against `bert-base-uncased`:

- `bert-base-uncased`:
  - `B1` acc `0.3812`, F1 `0.3081`
  - `D1` acc `0.4440`, F1 `0.4087`
  - `D2` acc `0.3902`, F1 `0.3289`
  - `D2.5` acc `0.3854`, F1 `0.3191`
  - `D3` acc `0.3721`, F1 `0.2877`
- `bert-large-uncased`:
  - `B1` acc `0.3656`, F1 `0.3654`
  - `D1` acc `0.4211`, F1 `0.4080`
  - `D2` acc `0.3687`, F1 `0.3135`
  - `D2.5` acc `0.3995`, F1 `0.3620`
  - `D3` acc `0.3947`, F1 `0.3931`

Takeaway from the base-vs-large uncased comparison:

- Scaling uncased BERT up helps `D2.5` and `D3` substantially.
- But scaling up does not help `D1`, and it does not make `D2` strong.
- So the effect of model size is method-dependent:
  - larger uncased BERT helps some hybrid / label-free-selector paths
  - but does not fix the core weakness of the pure `D2` path on the uncased family

Comparison against the earlier `bert-base-cased` fixed-test run:

- `bert-base-cased` still remains clearly stronger overall:
  - `D1` acc `0.4810`, F1 `0.4782`
  - `D2` acc `0.4621`, F1 `0.4574`
  - `D2.5` acc `0.4581`, F1 `0.4074`

Overall BERT-family read after this run:

- cased BERT remains preferable to uncased BERT for the current prototype method.
- `D1` is the most robust BERT-family method.
- `D2` on uncased BERT, even large, looks unreliable.
- `D2.5` on large uncased is interesting as a partial recovery, but still not a top-tier result compared with RoBERTa or cased BERT.

## 2026-04-04 RoBERTa-large fixed-test prototype run

Model:

- `FacebookAI/roberta-large`
- classifier: `prototype`
- `INCLUDE_D25=1`

Results:

- `B1 (raw)`:
  - test acc `0.3104`
  - test macro-F1 `0.2670`
- `D1 (debias_vl)`:
  - test acc `0.3478`
  - test macro-F1 `0.3048`
- `D2 (CBDC)`:
  - test acc `0.4072`
  - test macro-F1 `0.2262`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.4061`
  - test macro-F1 `0.2245`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.3098`
  - test macro-F1 `0.2664`

Critical interpretation:

- This is not evidence that "bigger model helps" in a simple way.
- `D2` and `D2.5` gain a lot of accuracy over the weak raw `B1` baseline, but they do so by almost collapsing onto the `neutral` class.
- The key warning sign:
  - `D2` neutral recall is `0.9622`
  - negative recall is `0.0110`
  - positive recall is `0.0471`
- So the higher accuracy is misleading:
  - macro-F1 is very poor
  - class balance is much worse than on `roberta-base`

Comparison against the earlier best `roberta-base` fixed-test run:

- `roberta-base`:
  - `B1` acc `0.4318`, F1 `0.3747`
  - `D1` acc `0.4709`, F1 `0.4577`
  - `D2` acc `0.4740`, F1 `0.4758`
  - `D2.5` acc `0.4544`, F1 `0.4411`
- `roberta-large`:
  - `B1` acc `0.3104`, F1 `0.2670`
  - `D1` acc `0.3478`, F1 `0.3048`
  - `D2` acc `0.4072`, F1 `0.2262`
  - `D2.5` acc `0.4061`, F1 `0.2245`

Takeaway:

- `roberta-large` is dramatically worse than `roberta-base` in the current prototype setup.
- Bigger RoBERTa did not improve the method.
- Instead, it appears to make the prototype geometry much less balanced, especially for the CBDC path.

Likely inference:

- The method is functioning technically on `roberta-large`, but the current prompt/prototype setup does not transfer cleanly to that backbone.
- Plausible reasons:
  - larger hidden space changes prototype geometry
  - the current one-layer tail adaptation may be too shallow for a much larger backbone
  - prompt embeddings may become less sentiment-balanced without additional calibration

Safe wording:

- "A larger backbone does not automatically help the current prototype CBDC pipeline; on `roberta-large`, higher D2 accuracy came with severe class-collapse and much worse macro-F1 than `roberta-base`."

Updated RoBERTa-family read:

- `roberta-base` remains the strongest backbone for the current CBDC-style prototype path.
- `roberta-large` should currently be treated as a negative or cautionary result rather than an upgrade.

## 2026-04-04 BERT-large-cased fixed-test prototype run

Model:

- `bert-large-cased`
- classifier: `prototype`
- `INCLUDE_D25=1`

Results:

- `B1 (raw)`:
  - test acc `0.3540`
  - test macro-F1 `0.2781`
- `D1 (debias_vl)`:
  - test acc `0.4335`
  - test macro-F1 `0.4082`
- `D2 (CBDC)`:
  - test acc `0.3560`
  - test macro-F1 `0.2825`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.3829`
  - test macro-F1 `0.3089`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.3868`
  - test macro-F1 `0.3296`

Interpretation:

- This is another case where scaling the backbone up does not help the pure `D2` path.
- `D1` is clearly the strongest result on this backbone.
- `D2` is almost identical to raw `B1`, despite Phase 2 training completing normally.
- `D2.5` and `D3` are both better than `D2`, which suggests the large cased BERT geometry is not interacting well with the current label-selected pure CBDC setup.

Important training clue:

- `D2` restored the best checkpoint at epoch `1/100`.
- That strongly suggests:
  - the optimization itself is active
  - but continuing CBDC tail training on this backbone tends to hurt the selector metric rather than improve it

Comparison against the earlier `bert-base-cased` fixed-test run:

- `bert-base-cased`:
  - `B1` acc `0.4287`, F1 `0.3785`
  - `D1` acc `0.4810`, F1 `0.4782`
  - `D2` acc `0.4621`, F1 `0.4574`
  - `D2.5` acc `0.4581`, F1 `0.4074`
- `bert-large-cased`:
  - `B1` acc `0.3540`, F1 `0.2781`
  - `D1` acc `0.4335`, F1 `0.4082`
  - `D2` acc `0.3560`, F1 `0.2825`
  - `D2.5` acc `0.3829`, F1 `0.3089`
  - `D3` acc `0.3868`, F1 `0.3296`

Takeaway from base-vs-large cased comparison:

- `bert-base-cased` is much stronger than `bert-large-cased` for every important method here.
- So larger cased BERT is not an upgrade for the current prototype pipeline.

Comparison against `bert-large-uncased`:

- `bert-large-uncased`:
  - `D1` acc `0.4211`, F1 `0.4080`
  - `D2` acc `0.3687`, F1 `0.3135`
  - `D2.5` acc `0.3995`, F1 `0.3620`
  - `D3` acc `0.3947`, F1 `0.3931`
- `bert-large-cased`:
  - `D1` acc `0.4335`, F1 `0.4082`
  - `D2` acc `0.3560`, F1 `0.2825`
  - `D2.5` acc `0.3829`, F1 `0.3089`
  - `D3` acc `0.3868`, F1 `0.3296`

Nuanced takeaway:

- At large size, cased BERT still wins for `D1`.
- But large uncased BERT is actually better than large cased BERT for `D2`, `D2.5`, and `D3`.
- So the simple story is not "cased always wins"; the safer story is:
  - `bert-base-cased` is the best BERT-family backbone tested so far
  - scaling BERT up, whether cased or uncased, does not help the current CBDC prototype setup

Updated BERT-family read:

- `bert-base-cased` remains the strongest BERT-family backbone overall.
- `D1` is still the most robust BERT-family method.
- The larger BERT models do not currently strengthen the pure `D2` story.
- If more BERT-family work is done, it should probably focus on prompt/bias calibration or deeper-tail variants rather than simply scaling model size.

## 2026-04-04 Qwen2.5-3B fixed-test prototype run

Model:

- `Qwen/Qwen2.5-3B`
- classifier: `prototype`
- `INCLUDE_D25=1`

Results:

- `B1 (raw)`:
  - test acc `0.4049`
  - test macro-F1 `0.2058`
- `D1 (debias_vl)`:
  - test acc `0.3953`
  - test macro-F1 `0.2889`
- `D2 (CBDC)`:
  - test acc `0.4049`
  - test macro-F1 `0.2192`
- `D2.5 (CBDC no-label-select)`:
  - test acc `0.4027`
  - test macro-F1 `0.2186`
- `D3 (debias_vl->CBDC)`:
  - test acc `0.4035`
  - test macro-F1 `0.2389`

Critical interpretation:

- Yes, this is a form of class-collapse, but not a numerical failure or NaN-style "zeroing".
- The collapse is mostly toward `neutral`.
- The most important nuance is:
  - the raw baseline `B1` is already collapsed
  - `negative` recall is `0.0000` even before debiasing
  - `neutral` recall is `0.9832`
- So the decoder-family issue is not that `D2` suddenly broke a healthy backbone.
- Rather:
  - the raw Qwen prototype geometry is already badly miscalibrated for this task
  - CBDC training is active, but it is operating on top of an already collapsed prototype setup

What Phase 2 says:

- `D2` training is definitely doing something:
  - `ck` falls from `7.8405` to `0.0012`
  - `selector_f1` rises from `0.4065` to `0.4446`
- So the method is not inert on Qwen.
- The mismatch is between:
  - learned representation changes
  - and the final prototype-classification geometry

Method comparison on Qwen:

- `D1` gives the best macro-F1 on this backbone.
- `D3` is second-best on macro-F1.
- `D2` and `D2.5` are only marginally above raw on macro-F1 and essentially tied on accuracy.
- This suggests:
  - pure CBDC is not enough to repair the decoder-family prototype collapse under the current prompt setup
  - DebiasVL-style projection still helps more than CBDC here

Safe wording:

- "Qwen2.5-3B shows severe neutral-class prototype collapse already at the raw baseline. CBDC training remains active and reduces the bias-alignment loss strongly, but the final prompt/prototype geometry stays poorly calibrated for sentiment classification."

Likely inference:

- For decoder-family models, the current prototype prompts and pooling choice are probably the bottleneck.
- Good next tests, if decoder models are pursued further:
  - try `--text_unit tweet`
  - rethink class prompts for decoder LMs
  - test alternative decoder pooling beyond last-token only

## 2026-04-04 Llama-3.2-3B access failure

Model:

- `meta-llama/Llama-3.2-3B`

Outcome:

- The run failed at Phase 1 with a Hugging Face gated-repo authentication error.
- This is an access / authentication issue, not a method or implementation failure.

Important note:

- The failure is actually desirable behavior compared with the old code path, because the pipeline now fails loudly instead of silently falling back to a different model.

## 2026-04-04 Prompt-bank bottleneck hypothesis

Current strongest hypothesis:

- `prompts.py` is now a major bottleneck, especially for:
  - large backbones
  - decoder-family models
  - the pure CBDC path (`D2`, `D2.5`)

Why this hypothesis is plausible:

1. Raw prototype collapse already appears before CBDC training on some backbones

- Example: `Qwen/Qwen2.5-3B`
  - `B1` is already heavily collapsed toward `neutral`
  - so the issue cannot be blamed only on CBDC training
- Since `B1` uses raw class prompt prototypes, this points directly at prompt / prototype geometry.

2. `D2` and `D2.5` often fail in similar ways on the hard backbones

- Example: `roberta-large`
  - `D2` and `D2.5` are almost identical
  - both drift toward severe neutral-class collapse
- This suggests the problem is upstream of checkpoint selection.
- That makes prompt design a stronger suspect than the `D2` vs `D2.5` selector difference.

3. The fixed CBDC bank contains anchors that are not purely nuisance features

- `CBDC_STYLE_BIAS_PAIRS` currently includes:
  - repeated question marks
  - exclamation marks
  - very short text
  - laughter words
  - emoticons
- In sentiment data, several of these are genuine sentiment carriers, not just spurious style.
- So the pure CBDC bank may be removing signal that actually helps classify sentiment.

4. `keep_text` may be too neutral-coded for a sentiment task

- The current `keep_text` prompts are generic social / daily-life updates.
- That is reasonable for preserving casual-message semantics.
- But in sentiment classification, preserving projections onto strongly neutral-looking prompts may bias the representation toward neutrality on some backbones.

5. `target_text` may be too artificial for some models

- Current examples:
  - `A negative-sentiment text.`
  - `A neutral-sentiment text.`
  - `A positive-sentiment text.`
- These are compact and useful for the current port, but they may be unnatural for decoder LMs and some larger encoders.

Safe interpretation:

- The method appears to be functioning technically.
- The next likely failure point is not the optimizer or the basic CBDC loop.
- It is the prompt bank:
  - class prompts
  - target prompts
  - keep prompts
  - fixed bias-pair prompts

Practical consequence:

- Future improvement should probably focus on prompt-bank redesign before further large-model scaling.
- Especially high-priority ideas:
  - test shorter / more natural class prompts
  - test `text_unit=\"tweet\"` for tweet-native and decoder models
  - replace riskier style-bias pairs with safer topic / structure pairs
  - redesign `keep_text` so it preserves content without over-encoding neutrality

## 2026-04-05 PCA diagnostic requested by PI

PI request:

- If the method relies on cosine similarity, then the vectors already encode similarity geometry.
- If those vectors are projected with PCA and colored by sentiment class, we should be able to see whether the representation actually moved.

What I checked in code:

- `pipeline/prototype_classify.py` normalizes both embeddings and class prompt prototypes, then scores them with a dot product.
- So the prototype evaluation stage is explicitly cosine-similarity based.
- `encoder.py` also returns normalized sentence embeddings for the encoder path.

Important conclusion:

- The project already had the right vectors for this diagnostic.
- It did **not** yet have the PCA visualization implemented.

Important design choice:

- A separate PCA per condition would be misleading, because each panel could rotate differently.
- To make "movement" interpretable, the PCA basis should be fit once on pooled normalized embeddings across conditions, then all conditions should be projected into that shared 2D basis.

Implementation added:

- Added `pipeline/plot_pca.py`.
- It:
  - loads cached split embeddings for `B1`, `D1`, `D2`, optional `D2.5`, and `D3`
  - normalizes embeddings before PCA so the view stays closer to the cosine-geometry used by prototype classification
  - fits one shared 2D PCA basis across the selected conditions
  - plots each condition in its own panel using the same coordinate system
  - colors points by sentiment class
  - overlays class centroids
  - overlays class prompt prototypes when available
  - overlays `B1` centroids as a reference so class-level movement is visible directly
  - saves both a PNG figure and a CSV of the projected coordinates

What the PCA is actually measuring:

1. Input objects being projected

- The plotted sample points are the cached sentence embeddings for a chosen split.
- These embeddings are L2-normalized before PCA.
- So each sample vector lies on or near the unit sphere in the original embedding space.
- The plotted prompt prototypes are also normalized before projection.

2. Relationship to cosine similarity

- The prototype classifier itself uses normalized vectors and dot products.
- For normalized vectors `u` and `v`:
  - `cos(u, v) = u · v`
  - `||u - v||^2 = 2 - 2 cos(u, v)`
- So in the full original space, cosine similarity and Euclidean distance are tightly linked.
- This is why PCA on normalized embeddings is a reasonable diagnostic for the cosine-based prototype geometry.

3. What PCA does mathematically

- PCA finds orthogonal directions of maximal variance after centering the pooled embedding cloud.
- `PC1` is the direction explaining the most variance.
- `PC2` is the next orthogonal direction explaining the next-most variance.
- `PC3` is the third such direction if requested.
- The coordinates shown in the PCA plot are just linear projections onto these directions.

4. Why the PCA basis is shared across conditions

- The basis is fit once on the pooled normalized embeddings from all selected conditions.
- That means each condition is shown in the same coordinate system.
- So if the cloud for `D2` appears shifted relative to `B1`, that is actual movement in a shared linear subspace, not an artifact of each panel being rotated separately.

5. What the different overlaid markers mean

- Sample points:
  - individual test examples
- Class centroids:
  - the mean projected location of all examples in a gold sentiment class for that condition
- Prompt prototypes:
  - the projected class prompt prototype vectors used by the prototype classifier
- Reference centroids:
  - the `B1` class centroids, overlaid on every panel so before/after movement can be seen directly

6. How to interpret movement

- Large centroid shift from `B1` to `D2` or `D3`:
  - the average class representation moved substantially along the dominant shared PCA directions
- Larger centroid separation between classes:
  - the class means are more separated in the retained low-dimensional subspace
- Smaller gap between a class centroid and its prompt prototype:
  - the prompt prototype is more aligned with the empirical class cloud in the retained PCA subspace

7. What PCA does *not* measure exactly

- PCA is not the classifier itself.
- It is only a low-rank linear view of the geometry.
- If `D2` looks close to `B1` in 2D, that does not prove the method had no effect.
- It may mean:
  - the effect lives in dimensions beyond `PC1` and `PC2`
  - the effect is nonlinear
  - or the effect is small in global variance terms but still relevant for class logits

8. Why 3D helps but still has limits

- `PC1`/`PC2` can miss class-relevant structure if the dominant variance is due to something else.
- Adding `PC3` can recover some of that hidden movement.
- But even 3D is still only a truncated linear projection, not the full embedding geometry.

9. Project-specific interpretation rule

- In this project, the most reliable reading is not:
  - "more visible separation in PCA always means better classifier performance"
- The safer reading is:
  - PCA tells us whether the dominant shared embedding geometry changed, in what direction it changed, and whether class means and prompt prototypes moved together or apart.
- So PCA is best treated as an explainability / representation-diagnostics tool, not as a replacement for accuracy or macro-F1.

Why this should be useful:

- It directly answers the PI's question about whether the representation space actually moved.
- It should also help diagnose current failure modes:
  - neutral collapse on some larger backbones
  - backbone-specific geometry differences
  - whether `D2` / `D2.5` move the class clouds in the same direction as their prompt prototypes

Smoke-test status:

- `python -m py_compile pipeline/plot_pca.py` passed.
- Synthetic cache smoke tests passed for:
  - full condition set including `D2.5`
  - cache without `D2.5`
  - subset plotting with `--no_prototypes`
  - subsampling via `--max_points_per_condition`
- The script successfully wrote both PNG and CSV outputs in each smoke test.

Useful bug caught during smoke testing:

- The first version relied indirectly on the `INCLUDE_D25` environment-sensitive artifact helper.
- That would have made `D2.5` discovery brittle in standalone plotting runs.
- The plotting script was updated to resolve condition slugs from its own local spec, so it now discovers cached `D2.5` outputs directly without depending on external environment flags.

Cluster runtime note:

- A later test on the login node failed before plotting started:
  - `ImportError: libtorch_cuda.so: failed to map segment from shared object`
- This is not a PCA-logic bug.
- It happens at `import torch`, before any cache loading or plotting.
- Since the cache is stored as `.pt`, the plotting script still needs a working PyTorch runtime even though the PCA itself is CPU-side.
- Practical workaround:
  - run the plotting command on an allocated compute node or inside the same Slurm environment used for training

## 2026-04-05 PCA result read and 3D extension

Current read from the generated 2D PCA CSV exports:

- In many models, the shared 2D PCA plane appears to be dominated more by condition-level movement than by clean class separation.
- `D1` often shows the largest centroid shift away from `B1` in the shared basis.
- `D2` and `D2.5` often remain much closer to `B1` in the 2D PCA plane, even when the downstream prototype metrics change.
- This means the 2D view is useful, but it should not be overinterpreted as a full summary of sentiment separability.

Practical implication:

- If class centroids still look nearly collapsed in 2D, that does not necessarily mean the method did nothing.
- It can also mean the dominant shared PCA axes are tracking other large sources of variance, while the task-relevant change lives partly outside the first two components.

Implementation follow-up:

- Extended `pipeline/plot_pca.py` with `--n_components 3`.
- The same shared-basis logic is preserved:
  - normalize embeddings
  - fit one PCA basis across selected conditions
  - project all conditions into that same basis
- The script now supports:
  - 2D output with `--n_components 2`
  - 3D output with `--n_components 3`
  - CSV export with `pc1`, `pc2`, and optional `pc3`

3D smoke-test result:

- A synthetic-cache 3D run completed successfully and wrote:
  - PNG output
  - CSV output with `pc1`, `pc2`, `pc3`
- Added a clearer cache-consistency check for prototype dimensionality mismatch, so inconsistent artifacts now fail with an informative error instead of a cryptic sklearn traceback.

## 2026-04-05 t-SNE / UMAP follow-up

PI concern:

- PCA axes may not be the right axes if the desired class bundling is nonlinear or not aligned with the dominant global variance directions.

Implementation follow-up:

- Extended `pipeline/plot_pca.py` to support:
  - `--method pca`
  - `--method tsne`
  - `--method umap`
- In the current environment:
  - `t-SNE` is available through `scikit-learn`
  - `UMAP` is not currently installed

Important interpretation note:

- `t-SNE` and `UMAP` are much better than PCA for showing local neighborhood bundling.
- So if the requirement is "show classes as bundled clouds", these methods are more likely to produce a visually satisfying plot.
- But they are also easier to overinterpret:
  - they preserve local neighborhoods better than global geometry
  - apparent cluster separation can look stronger than it really is in the original space

Safe recommendation:

- For explainability / honest geometry:
  - keep the shared PCA figures
- For a PI-facing "do the classes bundle up visually?" plot:
  - also generate `t-SNE` figures
- Best framing:
  - PCA = global linear movement diagnostic
  - t-SNE = local neighborhood / bundling diagnostic

t-SNE smoke-test status:

- A synthetic-cache `t-SNE` run completed successfully and wrote:
  - PNG output
  - CSV output
- So the nonlinear plotting path is working locally.

## 2026-04-05 Read of actual t-SNE results

After switching the `t-SNE` path to cosine distance and generating real plots on the cached model outputs, the overall pattern is still:

- class bundling exists only weakly
- most backbones do **not** show clean three-cluster sentiment separation

Interpretation:

- This makes the result more convincing, not less.
- If even cosine-based `t-SNE` does not produce dramatic class bundles, then the lack of clean separation is probably not just a PCA-axis artifact.
- It suggests the sentiment representation is genuinely entangled with other latent factors such as:
  - topic
  - style
  - backbone-specific geometry

High-level model read:

- `roberta`:
  - only mild local bundling
  - `D2` looks somewhat healthier than `B1`, but not cleanly separable
- `roberta-large`:
  - very weak separability
  - this is consistent with the earlier class-collapse / calibration concerns
- `bert`:
  - among the healthier cases
  - `D1` and `D3` show somewhat stronger local structure than raw
  - still not a clean three-cluster picture
- `qwen25_3b`:
  - remains entangled even in cosine `t-SNE`
  - so the decoder-family issue is not solved by changing visualization method
- `bertweet`:
  - mixed behavior
  - some views suggest more spacing, but local class purity is not consistently improved

Safe conclusion:

- `t-SNE` helps more than PCA for showing local neighborhood structure.
- But even `t-SNE` does not make the sentiment classes cleanly separable for most models.
- So the current evidence supports:
  - real representation movement
  - but only weak-to-moderate visual class separability
  - and substantial entanglement in the learned sentiment geometry

PI-safe wording:

- "We also tried cosine-based t-SNE to avoid the PCA-axis issue. The classes still do not separate cleanly for most backbones, which suggests the sentiment geometry is genuinely entangled rather than merely hidden by the wrong linear projection axes."

## 2026-04-05 PI follow-up: visualization semantics, checkpoint meaning, and benchmark framing

Recent PI questions focused less on whether the plotting code runs and more on what exactly is being visualized and whether the current task framing is the right one.

### 1. What the markers mean in the nonlinear plots

In the current plotting script:

- colored dots:
  - actual sample embeddings for the selected condition and split
- white circles:
  - `B1(raw)` reference centroids, overlaid only for comparison
- `X` markers:
  - current-condition class centroids
- stars:
  - class prompt prototypes

Important caveat:

- circles and `X` markers are overlays and do **not** define the t-SNE axes
- prompt prototypes are currently embedded jointly with the sample points when prototypes are enabled, so they can influence the final t-SNE map

So if a PI wants the cleanest possible "are the sample clouds themselves bundled?" figure, the safest follow-up is a sample-only cosine t-SNE with:

- no prototypes
- no class centroids
- no reference centroids

### 2. What vector is actually being visualized

The plots do **not** show all token vectors. They show the final sentence-level pooled embedding used by the pipeline.

- encoder-family models (`bert`, `roberta`, `xlm-roberta`, `distilbert`):
  - pooled from the first token of the final hidden state
- decoder-family models (`qwen2`, `llama`):
  - pooled from the last non-pad token hidden state

So the decoder-family answer is:

- yes, the visualization is using the pooled vector from the last non-pad token side

This matters for PI interpretation:

- the point cloud is a sentence-level representation cloud
- not a token-level geometric trajectory
- not the raw PGD perturbation vector itself

### 3. Why a `B1(raw)` centroid can appear outside a `D1` cloud in t-SNE

The white circle that appears on a `D1` plot is not the `D1` centroid. It is the `B1(raw)` reference centroid.

So if the `D1` cloud moved and the raw centroid stayed behind, seeing the circle outside the `D1` cloud is not inherently contradictory.

There is also a more subtle t-SNE issue:

- t-SNE is a nonlinear local-neighborhood visualization
- the centroid is computed after projection in the low-dimensional map
- for curved or crescent-shaped point clouds, the arithmetic mean can easily fall outside the visibly dense region

Safe interpretation rule:

- use PCA for centroid-shift / global-geometry discussion
- use t-SNE for local bundling / neighborhood discussion

### 4. Whether `D2/D2.5/D3` are real CBDC conditions or just PGD

They are not PGD-only ablations.

`D2`, `D2.5`, and `D3` are learned CBDC-style conditions:

- PGD is used inside the training loop to generate perturbation directions
- then the trainable tail is optimized with the CBDC losses
- then the full split is re-encoded using the resulting condition-specific encoder state

So the final visualization for `D2/D2.5/D3` is:

- not just "what happens if PGD is applied once"
- but the re-encoded embedding space after the CBDC-style training procedure

By contrast:

- `D1` is a DebiasVL-style projection condition, not an epoch-trained tail

### 5. How many epochs are trained, and what checkpoint is plotted

The current CBDC conditions train for up to `100` epochs.

The plotted condition embeddings are based on the selected best checkpoint, not always the final epoch.

- `D2`:
  - best epoch selected by labeled validation centroid macro-F1
- `D2.5`:
  - best epoch selected by lowest prompt loss only
- `D3`:
  - best epoch selected by labeled validation centroid macro-F1

Then the script re-encodes the split using that selected checkpoint and plots those embeddings.

So if the PI asks whether the plot is from the best validation checkpoint, the answer is:

- yes for `D2` and `D3`
- `D2.5` uses a label-free best-loss criterion instead of validation F1

`D1` does not have this best-epoch concept because it is a projection transform rather than a trained checkpointed tail.

### 6. Current task framing: not financial sentiment analysis

An important framing clarification for PI conversations:

- the current task is **not** financial sentiment analysis
- it is closer to general tweet / general sentiment classification with:
  - negative
  - neutral
  - positive

So if the discussion shifts to benchmark positioning, the more appropriate family is:

- TweetEval sentiment
- SemEval Twitter sentiment
- Sentiment140

rather than finance-specific sentiment benchmarks.

### 7. Current primary concern: `prompts.py`, not a dead optimizer

At this point, the strongest concern is not that the optimizer is doing nothing.

The runs already show:

- strong loss movement in Phase 2
- real centroid / geometry movement
- backbone-dependent class-collapse behavior

So the more plausible bottleneck is:

- prompt / prototype / bias-anchor geometry

In practice, this points back to:

- `cbdc/prompts.py`

as a major remaining source of mismatch, especially for:

- larger backbones
- decoder-family models
- cases where the representation clearly moves but the final sentiment calibration remains poor

Safe PI-facing summary:

- the method appears to be doing real latent-space work
- but the current prompt bank is likely under-calibrated for some backbone families
- so the main bottleneck now is prompt/prototype geometry rather than simple absence of learning

## 2026-04-09 Professor-facing brief generated

Created a concise-but-sufficient professor-facing project brief that is meant to explain:

- what the project is trying to do
- how the codebase is organized
- what each condition (`B1`, `D1`, `D2`, `D2.5`, `D3`) means
- what is actually being debiased
- how the CBDC training loop works in the local NLP adaptation
- what the current empirical interpretation is
- what the next practical steps are

Outputs:

- `report/cbdc_professor_brief.md`
- `report/cbdc_professor_brief.docx`

Important design choice:

- The document is written for "assignment-level but direction-setting" communication.
- It avoids overclaiming novelty.
- It also avoids stale hard-coded metric narratives and instead summarizes the current stable project understanding:
  - the method is active
  - the transfer is backbone-dependent
  - the main bottleneck is now prompt/prototype geometry calibration, especially in `cbdc/prompts.py`

## 2026-04-10 Gemma 4 support attempt

The codebase was extended to prepare for Gemma 4 support, but there is an important environment blocker.

### 1. Official model family status

From the official Google Gemma 4 model card and Hugging Face pages:

- Gemma 4 is now a real family, not just a placeholder idea.
- The main variants are:
  - `E2B`
  - `E4B`
  - `26B A4B`
  - `31B`
- The official docs describe them as multimodal decoder-style models with hybrid local/global attention and long context.

This matters for the local project because they are much closer to the existing decoder-family path (`qwen2`, `llama`) than to the encoder-family BERT/RoBERTa path.

### 2. Local code changes made

Added registry shortcuts in `config.py`:

- `gemma4-e2b`
- `gemma4-e2b-it`
- `gemma4-e4b`
- `gemma4-e4b-it`
- `gemma4-26b`
- `gemma4-26b-it`
- `gemma4-26b-a4b`
- `gemma4-26b-a4b-it`
- `gemma4-31b`
- `gemma4-31b-it`

Extended `encoder.py` so that:

- `gemma4` is treated as a decoder-family latent-tail model
- Gemma 4 names are recognized by the decoder-family detection logic
- the decoder tail uses the same last-non-pad-token pooling path as the current Qwen/Llama support
- sliding/global attention selection is generalized so non-Qwen decoder families can use the same infrastructure when they expose layer-level attention type metadata

Also added a Gemma-specific dtype policy:

- default load dtype is now `bfloat16` for Gemma 4
- other backbones stay on the previous `float32` default unless overridden with `MODEL_DTYPE`

Reason:

- without this, larger Gemma 4 variants would be unrealistic on a single GPU even if the architecture loaded successfully

### 3. Current blocker: installed Transformers build

The current environment can tokenize Gemma 4 but cannot load the model body.

Observed behavior:

- `AutoTokenizer.from_pretrained('google/gemma-4-E2B-it')` succeeds
- `AutoModel.from_pretrained(...)` fails because the installed Transformers build does not recognize `model_type='gemma4'`

The local code now raises a clearer error message explaining that:

- Gemma 4 needs a newer Transformers build on the H100 node before Phase 1 / Phase 2 can run

So the current state is:

- code path prepared
- registry added
- decoder-family adaptation logic added
- but actual Gemma 4 runs are blocked by library support, not by the project code anymore

### 4. H100 feasibility read

Using the official Gemma 4 model card as the rough size guide:

- `E2B`:
  - should be comfortably feasible on a 96 GB H100
- `E4B`:
  - should also be comfortably feasible on a 96 GB H100
- `26B A4B`:
  - plausible on a 96 GB H100 if loaded in `bfloat16`
  - not something the previous all-`float32` pipeline would safely support
- `31B`:
  - plausible but not yet "safe" enough to promise without a real run
  - the weights alone are large enough that this should be treated as a borderline case until validated in the actual cluster environment

So the practical recommendation is:

- first try `gemma4-e2b-it`
- then `gemma4-e4b-it`
- then, if the upgraded Transformers build works cleanly, test `gemma4-26b-it`
- treat `gemma4-31b-it` as experimental until memory is verified on the real H100 job

### 5. Relation to the current CBDC method

Conceptually, Gemma 4 fits the local CBDC adaptation reasonably well:

- frozen decoder backbone
- intermediate latent extraction
- perturbation injected before the final decoder block(s)
- last-token-side pooled embedding used for sentence-level representation

So if the environment issue is removed, the same decoder-family CBDC path should be the right first attempt for Gemma 4.

The remaining uncertainty is not the high-level method shape; it is:

- exact architecture support in the installed Transformers version
- whether Gemma 4's hybrid attention implementation exposes the same layer metadata that the generalized decoder-tail path expects

### Follow-up: practical support fix

The most realistic way to "fix Gemma 4 support" for this project was not a Kaggle-specific fallback loader, but upgrading the Transformers build itself.

Observed update:

- upgraded local environment from `transformers 5.3.0` to `transformers 5.5.3`
- after the upgrade:
  - `AutoConfig.from_pretrained('google/gemma-4-E2B-it')` succeeds
  - config type is now recognized as `Gemma4Config`
  - the real text decoder dimensions are exposed under `config.text_config`

Important architecture clarification:

- Gemma 4 is exposed through a multimodal wrapper config (`Gemma4Config`)
- the actual text decoder lives under:
  - `backbone.language_model`
- the real decoder hidden size and layer count live under:
  - `config.text_config.hidden_size`
  - `config.text_config.num_hidden_layers`

So the local encoder path was updated again to:

- use the multimodal wrapper only as the outer checkpoint container
- treat `backbone.language_model` as the true text core for:
  - hidden-size discovery
  - layer resolution
  - rotary embeddings
  - final norm
  - intermediate hidden-state extraction

This is a better fit than trying to run the entire multimodal wrapper as if it were a plain decoder-only text model.

Requirements update:

- `requirements.txt` now pins `transformers>=5.5.3`

Interpretation:

- for this project, upgrading Transformers is the right primary fix
- not writing a custom Kaggle-only Gemma 4 loader

Reason:

- the current CBDC port depends on:
  - tokenizer access
  - hidden-state extraction
  - partial-layer tail execution
  - architecture-aware causal masking

These are natural in a supported Transformers implementation, but would require a much larger custom integration effort in a separate runtime.

### Alternative runtimes: why they are not the main path here

Possible alternatives considered:

- Kaggle / Google Gemma examples
- Hugging Face collection
- official `google/gemma_pytorch` style routes

Current conclusion:

- Kaggle and Google docs are useful for weights and reference usage, but they are not the best direct fit for this project's latent-tail CBDC manipulation.
- A pure inference runtime such as vLLM or llama.cpp would also be a poor fit, because the project needs:
  - intermediate hidden states
  - tail-layer access
  - gradient-based latent perturbation

So the practical ranking is:

1. supported Transformers Gemma 4 path
2. only if that fails badly, consider a larger custom integration around a lower-level Gemma PyTorch implementation

At the moment, option 1 is clearly the right route.

### Regression check after Gemma 4 support changes

Follow-up task:

- include the Gemma 4 code-path changes in the journal
- verify that the refactor did not break previously working encoder / latent-tail paths

What changed in code:

- `config.py`
  - added Gemma 4 shortcut aliases:
    - `gemma4-e2b`, `gemma4-e2b-it`
    - `gemma4-e4b`, `gemma4-e4b-it`
    - `gemma4-26b`, `gemma4-26b-it`
    - `gemma4-26b-a4b`, `gemma4-26b-a4b-it`
    - `gemma4-31b`, `gemma4-31b-it`
- `encoder.py`
  - added `gemma4` to the supported decoder-family latent-tail types
  - introduced a `core_model` abstraction so the code can treat:
    - plain encoders normally
    - `Gemma4Config` checkpoints through `backbone.language_model`
  - generalized decoder tail execution so Llama / Qwen2 / Gemma4 all go through the same terminal-token latent-tail path
  - added dtype resolution:
    - Gemma 4 defaults to `bfloat16`
    - existing models remain `float32` unless `MODEL_DTYPE` is explicitly set
- `requirements.txt`
  - now requires `transformers>=5.5.3`

Sanity checks run:

- `python -m py_compile encoder.py config.py run_all.py data/embed.py cbdc/refine.py pipeline/prototype_classify.py pipeline/evaluate.py report/make_report.py`
  - passed
- registry resolution check:
  - `bert -> bert-base-cased`
  - `roberta -> roberta-base`
  - `distilbert -> distilbert-base-cased`
  - `qwen25-0.5b -> Qwen/Qwen2.5-0.5B`
  - `gemma4-e2b-it -> google/gemma-4-E2B-it`
- Gemma-capability check after upgrading Transformers:
  - local env now reports `transformers 5.5.3`
  - `AutoConfig.from_pretrained('google/gemma-4-E2B-it')` succeeds
  - `AutoConfig.from_pretrained('google/gemma-4-E4B-it')` succeeds
  - both expose `Gemma4Config`, with text-core dimensions under `config.text_config`

Runtime smoke test for old pipeline families:

- built tiny local models and ran the local `TransformerEncoder` against them with:
  - `encode_text(...)`
  - `get_intermediate_features(...)`
  - `encode_with_delta_from_hidden(...)`
- passed for:
  - `bert`
  - `roberta`
  - `distilbert`
  - `llama`
  - `qwen2`

Interpretation:

- the Gemma 4 refactor does **not** appear to have broken the previously supported encoder-family paths
- the existing decoder-family latent-tail logic for Llama and Qwen2 also still executes correctly after the refactor
- so the current code state is:
  - Gemma 4 support path added
  - older pipelines still smoke-test cleanly

Remaining caveat:

- this was a targeted smoke test, not a full re-run of all historical training jobs
- actual full Gemma 4 experiment runs still depend on the real cluster environment having:
  - updated `transformers`
  - enough VRAM for the selected Gemma variant

Current confidence level:

- high confidence that the code refactor itself did not regress the old architecture paths
- moderate confidence on full Gemma 4 execution until an actual H100 run is launched

### Gemma 4: reviving the `text_iccv` path more faithfully

Question raised:

- since Gemma 4 is multimodal and therefore closer to a VLM wrapper than a plain decoder LM, can the local `text_iccv` path be revived so it mirrors the original CBDC text-stage path more faithfully?

Short answer:

- yes, partially
- but the right fix was **not** "route through the vision tower"
- the right fix was to restore Gemma 4's missing text-layer side-channel inputs inside the latent-tail execution path

What was discovered from the installed Transformers Gemma 4 code:

- `Gemma4ForConditionalGeneration` is a multimodal wrapper with:
  - `model = Gemma4Model(config)`
  - `lm_head`
- `Gemma4Model` contains:
  - `language_model`
  - `vision_tower`
  - multimodal embedders for image/audio soft tokens
- for a text-only run, the actual sentence representation still comes from the text stack inside:
  - `backbone.language_model`

Important architectural nuance:

- Gemma 4 text is **not** just a plain "last token decoder" in the simple Llama/Qwen sense
- its text layers can consume:
  - `per_layer_input`
  - `shared_kv_states`
  - layer-type-specific rotary embeddings and attention masks
- the local first-pass Gemma support had treated the text core as an ordinary decoder tail
- that meant the manual latent-tail execution in `encode_with_delta_from_hidden(...)` was missing the per-layer side inputs that Gemma 4 uses internally

This matters specifically for `text_iccv`, because the whole Phase 2 method depends on:

- extracting an intermediate hidden state
- injecting `delta`
- manually executing the trainable tail block(s)

If that tail execution omits Gemma's extra inputs, the path is only a rough approximation.

What was changed to revive the path:

- `encoder.py`
  - `encode_with_delta_from_hidden(...)` now accepts `input_ids`
  - for `gemma4`, `_prepare_decoder_tail_inputs(...)` now reconstructs:
    - projected `per_layer_inputs`
    - layer-type-specific rotary embeddings
    - layer-type-specific attention routing
    - shared KV state container
  - the Gemma tail now feeds each decoder layer with:
    - `per_layer_input`
    - `shared_kv_states`
    - `position_embeddings[layer_type]`
    - `attention_mask[layer_type]`
- `cbdc/refine.py`
  - all `text_iccv` / `_pgd_bipolar` / class-prototype call sites now pass `input_ids` through to `encode_with_delta_from_hidden(...)`

Interpretation:

- this is a materially better match to Gemma 4's actual text path than the earlier generic decoder-only adaptation
- it is still not "CLIP text path unchanged", because Gemma 4 remains a generative multimodal decoder family, not a contrastive CLIP encoder
- but for the text-only CBDC adaptation, this is the correct architectural direction

Verification run:

- tiny synthetic `Gemma4Model` checkpoint created locally
- smoke test passed for:
  - `encode_text(...)`
  - `get_intermediate_features(...)`
  - `encode_with_delta_from_hidden(..., input_ids=...)`
- stronger smoke test also passed:
  - 1-epoch synthetic `text_iccv(...)` run on tiny Gemma 4 with `selector_mode='last'`
  - completed end-to-end and produced:
    - final directions tensor
    - final class prototype tensor

Also rechecked after this patch:

- tiny `qwen2` latent-tail smoke test still passes

Best wording:

- Gemma 4 being multimodal did **not** mean "use the full VLM path with vision features" for this project
- it meant the text-side latent-tail path needed to respect Gemma's richer text-layer mechanics
- after this patch, the local `text_iccv` path is revived for Gemma 4 in the way that is most faithful to the current text-only CBDC adaptation

### CLIP text path vs current NLP / Gemma path

Important clarification:

- the remembered CLIP sketch:

  - token embedding
  - positional embedding
  - all transformer blocks frozen
  - `ln_final`
  - `[token_max]`
  - `text_projection`
  - normalize

  is closer to the original code's `proj` mode than to its default CBDC text-training path

What the original CLIP code actually defaults to:

- `train_bafa.py` sets:
  - `txt_learn_mode="linear"`
- `SetTarget(..., modal='txt', learn_mode='linear')` then:
  - freezes the whole text tower
  - unfreezes the last text transformer block
  - keeps `ln_final` and `text_projection` as the fixed readout

So the original default text-stage CBDC shape is better summarized as:

- frozen token embedding / positional embedding / early text blocks
- extract intermediate text latent
- apply PGD in latent space
- run the last text transformer block (trainable)
- `ln_final`
- `[token_max]`
- fixed `text_projection`
- normalize

That is closer to the local NLP adaptation than the projection-only memory sketch.

Current local NLP/Gemma path:

- encoder-style models:
  - frozen early layers
  - latent perturbation before the final layer
  - trainable last encoder block
  - pooled sentence representation
  - normalize
- decoder-style models:
  - frozen early decoder stack
  - latent perturbation at the terminal pooled-token position
  - trainable last decoder block
  - final norm
  - last non-pad token pooling
  - normalize

So the safest comparison remains:

- original CBDC default text path:
  - frozen body + perturbed latent + trainable last text block
- current NLP/Gemma path:
  - frozen body + perturbed latent + trainable last text/decoder block

### Gemma 4 decoder-layer training feasibility

Question checked:

- is decoder-layer training actually possible for Gemma 4, given:
  - RoPE / p-RoPE style rotary handling
  - hybrid sliding/global attention
  - per-layer embeddings in the small dense models
  - the `26B A4B` variant naming, which could be confused with a quantized format

Architecture reading:

- from the official Gemma 4 Hugging Face model card:
  - Gemma 4 uses hybrid attention
  - small dense models (`E2B`, `E4B`) use Per-Layer Embeddings (PLE)
  - `26B A4B` is a Mixture-of-Experts model where "A" means **active parameters**
  - it is **not** a "TurboQuant" style quantized checkpoint name

Current interpretation:

- `A4B` = active-parameter MoE naming
- not a reason by itself that tail-layer training would be impossible

What was verified locally from the real released configs:

- `google/gemma-4-E2B-it`
  - `num_kv_shared_layers = 20`
  - `hidden_size_per_layer_input = 256`
- `google/gemma-4-E4B-it`
  - `num_kv_shared_layers = 18`
  - `hidden_size_per_layer_input = 256`
- `google/gemma-4-26B-A4B-it`
  - `num_kv_shared_layers = 0`
  - `enable_moe_block = True`
  - `hidden_size_per_layer_input = 0`
- `google/gemma-4-31B-it`
  - `num_kv_shared_layers = 0`
  - `hidden_size_per_layer_input = 0`

This split matters:

- `E2B / E4B`
  - need the local tail path to carry:
    - per-layer inputs
    - shared-KV state
    - layer-type-specific rotary embeddings / masks
  - a pure one-layer tail starting only at the final layer is **not** enough
  - the final layer is shared-KV and needs the provider layer to run first
- `26B A4B / 31B`
  - do **not** use the same shared-KV / PLE setup at the tail
  - so the local last-block decoder-tail assumption is much closer to adequate

Additional patch made:

- `encoder.py`
  - now computes an architecture-aware default Gemma tail start layer
  - for shared-KV Gemma variants, it backs up to the provider layer needed to populate the final layer's shared KV state
  - this keeps only the last layer trainable, but runs enough frozen tail context to make the final layer executable

Concrete check:

- synthetic tiny Gemma 4 config built with:
  - hybrid attention
  - per-layer inputs
  - shared-KV
  - MoE block enabled
- after the tail-start fix:
  - backward pass through `encode_with_delta_from_hidden(...)` succeeds
  - gradients reach:
    - the last decoder layer parameters
    - the injected `delta`

Interpretation:

- yes, decoder-layer training is genuinely possible
- RoPE / hybrid attention / MoE do not make the method impossible
- but:
  - small dense Gemma (`E2B`, `E4B`) needs a deeper architecture-aware tail start
  - `26B A4B` and `31B` are structurally simpler for this specific text-only latent-tail adaptation

Current best practical recommendation:

- if the goal is "most faithful Gemma 4 CBDC run with least architectural friction":
  - `gemma4-26b-it` or `gemma4-31b-it` are conceptually cleaner targets
- if the goal is "cheapest first Gemma experiment":
  - `gemma4-e2b-it` is still viable, but its tail path is more architecture-sensitive because of shared-KV + PLE

Support-scope decision:

- there is no need to present all Gemma 4 variants as equally supported
- the practical support story should be intentionally narrower:
  - primary recommended Gemma target: `gemma4-26b-it`
  - no need to stretch the codebase to claim all Gemma 4 variants
  - small `E2B/E4B` variants are intentionally out of scope in the cleaned-up path

Implementation note:

- the registry and code were cleaned up accordingly:
  - keep the `26B A4B` aliases
  - remove the extra small/large alias clutter that is not needed for the current experiments
  - remove the small-Gemma-specific latent-tail machinery that was only there to support `E2B/E4B`

### Gemma support cleanup

Follow-up cleanup decision:

- simplify the Gemma integration to only what is needed for:
  - existing pipelines
  - current Gemma experiments

Code cleanup made:

- `config.py`
  - trimmed Gemma aliases down to the `26B A4B` path:
    - `gemma4-26b`
    - `gemma4-26b-it`
- `run_all.py`
  - example usage now points only to:
    - `gemma4-26b-it`
- `encoder.py`
  - removed the small-Gemma-specific handling for:
    - reconstructed `per_layer_inputs`
    - deeper shared-KV tail-start fallback
  - kept only the necessary Gemma 4 text-core logic for the cleaner 26B-style path:
    - resolve `backbone.language_model`
    - layer-type-specific rotary embeddings
    - sliding/full attention routing
  - added an explicit guard:
    - Gemma 4 variants with per-layer inputs or shared-KV text layers now fail early with a clear `NotImplementedError`
- `cbdc/refine.py`
  - removed the extra `input_ids` plumbing that had only been introduced for the small-Gemma path

Interpretation:

- this makes the Gemma integration smaller and easier to reason about
- it avoids leaving half-supported special cases in the code
- it keeps the path that is actually relevant to the current experiments

Verification after cleanup:

- `python -m py_compile encoder.py cbdc/refine.py config.py run_all.py`
  - passed
- tiny smoke tests passed for existing pipelines:
  - `bert`
  - `qwen2`
- tiny `26B/31B`-style Gemma smoke test passed:
  - no per-layer inputs
  - no shared-KV text layers
- tiny 1-epoch `text_iccv(...)` run on cleaned Gemma path passed end-to-end
- tiny `E2B/E4B`-style Gemma now fails early with the intended message:
  - current support is intentionally scoped to `26B/31B`-style text variants

Current project-facing wording:

- Gemma 4 support is intentionally narrow
- the maintained path is the `26B A4B` CBDC experiment path
- existing non-Gemma pipelines remain intact after the cleanup

## 2026-04-10 PI suggestion: pair-free hard-example PGD branch

PI suggestion:

- since this project is text-only, instead of hand-constructing bias pairs and prompt poles, identify samples that are currently misclassified
- run PGD on those samples to make the classification loss larger
- then train the model/tail to classify those hard adversarial samples correctly
- in that setup, explicit bias-pair construction may no longer be necessary

Korean paraphrase of the suggestion:

- "텍스트로만 하니까 오분류되는 케이스만 PGD 적용해서"
- "loss가 커지도록 샘플 찾아서"
- "그걸 정분류로 학습시키면"
- "굳이 페어 안 만들어도 될지도"

Current interpretation:

- yes, this is a plausible new branch
- it is **not** the same as current `D2/D2.5/D3`
- it would be closer to:
  - hard-example adversarial augmentation
  - or a text analogue of BiasAdv-style bias-conflicting augmentation
- the appeal is that it may avoid the current `prompts.py` bottleneck
  - especially if the main failure mode is poor prompt/prototype/bias-anchor calibration

Relation to current code:

- current CBDC path in `cbdc/refine.py`
  - builds bias directions from `bias_anchors` / `anti_anchors`
  - uses `_pgd_bipolar(...)` to push latent perturbations toward the handcrafted bias poles while preserving semantics
  - then trains the last tail layer using `match_loss + ck_loss`
- the PI suggestion would replace that logic with something sample-centric:
  - use labeled samples
  - find currently misclassified or low-margin examples under the current sentiment decision rule
  - apply PGD in latent space to **increase classification loss**
  - then update the tail to classify those adversarial hard examples correctly

Natural implementation path in this repo:

- reuse the existing latent-tail machinery:
  - `encoder.get_intermediate_features(...)`
  - `encoder.encode_with_delta_from_hidden(...)`
- replace `_pgd_bipolar(...)` with a new PGD routine that maximizes:
  - prototype cross-entropy loss on the true sentiment label
  - or centroid/prototype margin loss
- select hard examples using one of:
  - actual misclassification under current prototypes
  - low-margin samples even if still correctly classified
  - disagreement cases already exported by `pipeline/prototype_classify.py`
- then train the tail on:
  - clean sample + adversarial sample consistency
  - or directly on adversarially perturbed examples with the true label

Why this is interesting:

- it removes the need to manually specify bias poles such as punctuation / emoticons / length cues
- it directly targets the model's own failure cases
- if prompt/bias engineering is the main bottleneck, this could be a cleaner branch than continuing to tune `prompts.py`

Why this is **not** automatically debiasing:

- plain adversarial training on misclassified samples can improve robustness without actually removing spurious bias
- to make the branch genuinely "debiasing", we still need a reason to believe the attacked failures are shortcut-driven rather than arbitrary hard cases
- in BiasAdv terms, the stronger version would use:
  - an auxiliary biased predictor
  - and generate perturbations that attack the biased decision while preserving the debiased decision
- the simpler first NLP version could still be worthwhile as a practical assignment-scale experiment, but should be framed carefully

Practical recommendation:

- treat this as a new method branch, e.g. `D4`
- do **not** redefine current CBDC as this method
- first version can be intentionally simple:
  - prototype-based sentiment logits
  - misclassified / low-margin training subset
  - latent PGD maximizing true-label CE loss
  - tail-layer adversarial fine-tuning to recover correct sentiment classification

Most likely benefit:

- if the current obstacle is really `prompts.py`, this branch may work better because it relies less on handcrafted bias phrasing

Most likely risk:

- if we only chase ordinary misclassification, the method may become generic adversarial hard-example training rather than a clear debiasing method

Best framing at this point:

- "worth trying as a separate experimental branch"
- "closer to pair-free adversarial hard-example debiasing / augmentation"
- "promising mainly because it may bypass the fragile prompt-pair design in the current CBDC adaptation"

Clarification after re-reading the PI suggestion:

- the suggestion can still be seen as **CBDC-style in mechanics**
  - latent PGD
  - frozen body + trainable tail
  - adversarially generated representation shift
- but it is **not CBDC-style in supervision source**
  - current `D2/D2.5/D3` mostly use handcrafted prompt banks, bias poles, and prompt/prototype geometry
  - the PI suggestion uses actual sentiment classification failures as the driver for PGD and tail updates

So the clean interpretation is:

- "CBDC-shaped training skeleton"
- but with a different objective
- namely:
  - move from pair-defined bias perturbation
  - to label-guided hard-example adversarial correction

Supervision note:

- yes, as stated, this new branch would be **more clearly supervised**
- why:
  - it explicitly depends on whether a sample is misclassified
  - it needs a true label or at least a trusted target label to define "loss should get larger" and "train it back to correct classification"
- compared to current methods:
  - `D1` is not supervised fine-tuning
  - `D2` is prompt-driven training with labeled validation checkpoint selection
  - `D2.5` is the most label-light current variant because checkpoint selection is loss-only
  - the proposed hard-example PGD branch would use labels directly in the inner/outer training objective

Best wording:

- not "fully unsupervised debiasing"
- not "pure prompt debiasing"
- more like:
  - supervised adversarial hard-example debiasing
  - or supervised latent-space augmentation with a CBDC-style tail path

## 2026-04-10 BiasAdv notes for future reference

Paper:

- **BiasAdv: Bias-Adversarial Augmentation for Model Debiasing**
- Jongin Lim, Youngdong Kim, Byungjai Kim, Chanho Ahn, Jinwoo Shin, Eunho Yang, Seungju Han
- CVPR 2023

Core idea:

- train an **auxiliary biased model**
- use adversarial attack on that biased model to generate synthetic **bias-conflicting** samples
- train the debiased model on those synthetic samples so it relies less on spurious shortcuts

Important mechanism from the paper:

- BiasAdv is not just generic adversarial training
- it has two roles:
  - attack the prediction of the **biased auxiliary model**
  - while preserving the prediction of the **debiased model**
- this is the key reason the paper can claim the augmentation is "bias-conflicting" rather than merely noisy or hard

Why that matters for our NLP branch:

- if we only do:
  - "find misclassified samples"
  - "use PGD to increase true-label loss"
  - "train on them"
- then the branch is closer to adversarial hard-example training / robustness training
- to make it more BiasAdv-like, we would ideally need:
  - a deliberately shortcut-biased auxiliary sentiment predictor
  - and an objective that attacks the shortcut-biased decision while preserving the intended sentiment signal

Practical translation to our repo:

- simplest first branch:
  - use current prototype sentiment classifier
  - identify misclassified or low-margin labeled samples
  - perform latent PGD to increase true-label sentiment loss
  - update the tail to recover correct classification
- stronger BiasAdv-style branch:
  - create an intentionally biased auxiliary prototype rule or model
  - generate latent perturbations that specifically break that biased predictor
  - keep the debiased sentiment decision stable

Connection to current concerns:

- this is attractive because it may reduce dependence on `prompts.py`
- but if we want to call it a debiasing method rather than generic adversarial fine-tuning, the BiasAdv distinction above is important

Safe summary:

- PI suggestion: plausible and potentially useful
- method label if implemented naively:
  - supervised adversarial hard-example branch
- method label if implemented in a BiasAdv-like way:
  - pair-free adversarial debiasing branch with an auxiliary biased predictor

BiasAdv intuition, phrased in the PI's words:

- "오분류를 하는 이유를 알아서 찾아라"

Important clarification:

- this does **not** mean the method produces a human-readable explanation by itself
- it means:
  - do not hand-specify the bias type in advance
  - let the adversarial optimization discover the direction / perturbation that most exposes the shortcut the current model is relying on

So the implicit logic is:

- if the model is misclassifying because of a spurious shortcut
- and we attack the model in a way that targets that shortcut-sensitive decision rule
- then the resulting adversarial example / latent delta is a proxy for
  - "the reason this sample was being misclassified"
  - or more carefully:
  - "the shortcut-sensitive direction that explains the current error"

Why this is attractive for our NLP setting:

- right now we manually guess possible shortcuts in `prompts.py`
  - punctuation
  - emoticons
  - short length
  - laughter tokens
  - style/register cues
- the PI's suggestion is more like:
  - stop guessing all of them by hand
  - let the model's own adversarial failure mode reveal which direction matters

Best careful wording:

- not "the model truly knows the semantic reason in an interpretable symbolic sense"
- but:
  - "the attack discovers an error-inducing latent direction"
  - "which can serve as a proxy for the shortcut feature causing the misclassification"

This is the strongest conceptual appeal of the BiasAdv-style branch for this project:

- it may replace manually engineered bias-pair design with model-discovered shortcut directions

## 2026-04-10 Implemented D4: adversarial discovery -> CBDC

Implementation decision:

- keep `D3` unchanged as the existing:
  - `DebiasVL/topic-mining discovery -> CBDC`
- add a new optional condition:
  - `D4 (adv-discovery->CBDC)`

Interpretation:

- this follows the stronger PI-aligned reading:
  - use adversarial failure to **discover** shortcut-sensitive latent directions
  - then feed those discovered directions into the existing CBDC `text_iccv(...)` stage
- so this is **not** plain adversarial fine-tuning replacing CBDC
- it is:
  - adversarial bias-direction discovery
  - followed by ordinary CBDC tail training

Implemented path:

1. Build raw class prompt prototypes from the CBDC class prompts.
2. Score cached raw train embeddings with those prototypes.
3. Select hard samples per class:
   - misclassified first
   - then lowest-margin correct samples as fallback
4. Run latent PGD on those selected samples to maximize prototype CE loss,
   while lightly preserving the original semantics through the existing keep-loss.
5. Collect the resulting embedding-space shifts:
   - `z_adv - z_orig`
6. SVD that delta bank.
7. Use the top singular directions as:
   - `bias_anchors`
   - and their sign-flipped copies as `anti_anchors`
8. Feed those anchors into the existing `text_iccv(...)` loop.

What changed in code:

- `cbdc/refine.py`
  - added `D4_LABEL`
  - added hard-example prototype scoring helper
  - added balanced hard-sample selection helper
  - added D4-specific PGD discovery routine
  - added SVD-based direction extraction for the discovered delta bank
  - added `discover_adv_bias_map(...)`
  - added optional `D4` materialization branch in `main()`
- `config.py`
  - added D4 discovery hyperparameters:
    - `d4_max_hard_per_class`
    - `d4_num_samples`
    - `d4_ce_temperature`
- `pipeline/artifacts.py`
  - `D4` is now included when `INCLUDE_D4=1`
- `run_all.py`
  - added `--include_d4`
- `pipeline/evaluate.py`
  - comparative analysis now reports `D4` when present
- `pipeline/plot_pca.py`
  - plotting can include `D4` when `INCLUDE_D4=1`

Important scope choice:

- `D4` is opt-in
- existing `B1 / D1 / D2 / D2.5 / D3` runs stay unchanged unless `INCLUDE_D4=1`
- this keeps old comparisons stable and preserves `D3` as the DebiasVL ablation

Verification:

- `python -m py_compile` passed for:
  - `cbdc/refine.py`
  - `pipeline/artifacts.py`
  - `pipeline/evaluate.py`
  - `pipeline/classify.py`
  - `pipeline/prototype_classify.py`
  - `pipeline/plot_pca.py`
  - `run_all.py`
  - `config.py`
- tiny end-to-end D4 smoke test passed:
  - temporary tiny BERT model
  - temporary cached raw train/val/test splits
  - `discover_adv_bias_map(...)`
  - `materialize_cbdc_condition(D4, ...)`
  - condition splits successfully written under:
    - `conditions/d4_adv_discovery_cbdc/`
- post-change decoder-family regression smoke tests still passed for:
  - `qwen2`
  - `llama`

Practical run command:

- `python run_all.py --classifier prototype --include_d4`
- optionally with:
  - `--include_d25`
  - and a model shortcut such as `--model roberta`

Current caveat:

- the present D4 discovery step is still a **simple first version**
- it attacks the current prototype sentiment decision directly
- so it is best described as:
  - adversarial shortcut-direction discovery for CBDC
- not yet a full BiasAdv replica with a separate explicitly biased auxiliary predictor

## 2026-04-10 BiasAdv methodology reference (Lim et al., CVPR 2023)

Paper: "BiasAdv: Bias-Adversarial Augmentation for Model Debiasing"

### Core idea

- models trained on biased data learn spurious shortcuts (e.g. topic predicts sentiment)
- re-weighting alone overfits because bias-conflicting samples are too scarce
- BiasAdv generates **synthetic bias-conflicting samples** via adversarial attack on a biased auxiliary model

### Two-model setup

1. **Auxiliary biased model g_φ**: intentionally trained to learn shortcuts
   - trained with GCE loss so it latches onto easy/spurious correlations
   - in our NLP context: a projector/head that predicts sentiment via topic shortcuts
2. **Debiased model f_θ**: the model we actually want to deploy
   - trained to make correct sentiment decisions independent of topic

### Adversarial sample generation (Eq. 3)

```
x_adv = argmax_{x̃ := x + ε} [ L(x̃, y; φ) − λ · L(x̃, y; θ) ]
```

- first term `L(x̃, y; φ)`: **attack** the biased model's prediction (flip its shortcut-based decision)
- second term `−λ · L(x̃, y; θ)`: **preserve** the debiased model's prediction (keep the correct class)
- uses PGD to solve this; perturbation alters the bias cue while keeping intrinsic class attributes intact
- λ > 0 is a hyperparameter controlling preservation strength

### Training objective (Eq. 4)

```
R_a(θ) = E_{(x,y)~D} [ ω_x · L(x, y; θ) + ω_adv · L(x_adv, y; θ) ]
```

- ω_x = W(x, y; θ, φ) from any existing re-weighting method (LfF, JTT, etc.)
- ω_adv = β · (1 − ω_x): adversarial weight complements the re-weighting
- β controls importance of adversarial augmented data
- intuition: bias-guiding samples that get down-weighted by re-weighting are instead translated into synthetic bias-conflicting samples via PGD

### Key design choices from the paper

- auxiliary batch normalization for adversarial images (different distribution from clean)
- works as a plug-in on top of any re-weighting method (ERM, LfF, JTT)
- no bias annotations or prior knowledge of bias type required
- hyperparameters per dataset:
  - λ ∈ {0.5, 1.0}, ε ∈ {0.3, 0.5, 0.7}, PGD steps S ∈ {3, 5, 7}, β ∈ {0.5, 1.0, 1.5}

### Why it works (paper's analysis)

1. **Decision boundary extension**: adversarial samples land near the biased model's decision boundary; since bias attributes are the easiest shortcut, PGD preferentially attacks the bias cue, producing synthetic bias-conflicting samples
2. **Sample diversity**: most training data is bias-guiding, so x_adv is translated from abundant bias-guiding samples, leveraging their diverse intrinsic attributes
3. **Prevents overfitting**: unlike pure re-weighting which over-fits on the few real bias-conflicting samples, BiasAdv provides unlimited synthetic ones
4. **Preserves bias-guiding performance**: the λ regularization term prevents intrinsic attributes from being compromised

### Ablation findings

- random noise augmentation: marginal improvement
- AdvProp (attacking f_θ instead of g_φ): **degrades** performance — attacking the debiased model hurts
- BiasAdv (λ=0, no preservation term): significant improvement but less than full BiasAdv
- full BiasAdv: best results — the preservation term ensures only bias cue is attacked

### Relation to current D4 and the PI's suggestion

The PI's suggestion maps directly to BiasAdv's two-model formulation:

- **편향된 projector** (biased) = g_φ: an auxiliary head deliberately trained on shortcuts
  - in our case: a sentiment head that learns topic→sentiment correlations
- **비편향된 projector** (debiased) = f_θ: the head we want to deploy
  - in our case: a sentiment head that classifies based on genuine sentiment cues

The PI's flow:
1. start with existing BERT (treat it as biased baseline)
2. train a new BERT/head to be debiased (like debias_vl initial pass)
3. then apply BiasAdv-style PGD: attack the biased head, preserve the debiased head
4. train on the resulting adversarial samples with correct labels
5. this avoids needing to manually construct bias pairs

### Gap between current D4 and full BiasAdv

Current D4:
- uses **one model** (prototype classifier) — attacks its own sentiment decision
- uses PGD deltas for **direction discovery** (SVD → anchors), then feeds into CBDC tail training
- no separate biased auxiliary predictor

Full BiasAdv adaptation would require:
1. train an auxiliary biased head g_φ (e.g. GCE-trained sentiment head that latches onto topic)
2. keep the debiased head f_θ (the one being improved)
3. PGD in embedding space: maximize L(z̃, y; φ) − λ · L(z̃, y; θ)
4. train f_θ on both clean and adversarial embeddings with correct sentiment labels
5. optionally combine with re-weighting (ω_x / ω_adv trade-off)
