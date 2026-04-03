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
