# Sentiment Pipeline Guide

This repository now has one shared sentiment pipeline spine with two entry paths built on top of it:

1. The original CBDC path for confound-direction discovery and debiasing.
2. A transformer fine-tuning path for direct sentiment classification.

The important design choice is that both paths reuse the same core pieces instead of maintaining separate stacks.

## What This Repo Does

At a high level, the codebase is organized around:

1. Loading and standardizing a 3-way sentiment dataset.
2. Encoding text and caching reusable artifacts.
3. Optionally running CBDC and projection-based debiasing.
4. Training a classifier.
5. Producing a report.

The shared spine is:

`dataset.py -> data/embed.py -> cached artifacts -> pipeline/classify.py -> pipeline/evaluate.py`

CBDC is built on top of that shared spine, not separate from it.

## Which Entry Script To Use

Use these from inside `project/`.

| Goal | Script |
|---|---|
| Full original CBDC pipeline | `python run_all.py` |
| Run the original pipeline but skip CBDC training | `python run_all.py --skip_cbdc` |
| Fine-tune a transformer classifier directly | `python run_transformer.py` |
| Debug phases more easily in-process | `python run_all_debug.py --inprocess` |
| Run a single original phase | `python run_all.py --only_phase N` |
| Resume original pipeline from a phase | `python run_all.py --start_phase N` |
| Run a single transformer phase | `python run_transformer.py --only_phase N` |
| Resume transformer pipeline from a phase | `python run_transformer.py --start_phase N` |

## Quick Start

```bash
cd project
python -m pip install -r requirements.txt
```

### Original CBDC pipeline

```bash
python run_all.py
```

Useful variants:

```bash
python run_all.py --skip_cbdc
python run_all.py --model bertweet
python run_all.py --start_phase 2
python run_all.py --only_phase 4
python run_all.py --no_sent_orthogonal_pgd
```

### Transformer classifier pipeline

```bash
python run_transformer.py
```

Common variants:

```bash
python run_transformer.py --model distilbert --epochs 3 --unfreeze_layers 2
python run_transformer.py --model finbert --max_length 160
python run_transformer.py --model distilbert --input_mode text_selected_pair --head_type mlp --loss_name focal
python run_transformer.py --model distilbert --use_time_of_tweet --use_age_of_user
python run_transformer.py --only_phase 2
python run_transformer.py --start_phase 2
```

## Mental Model Of The Code

The code is not coordinated by one large pipeline class. It is coordinated by small reusable modules and runner scripts.

Think of it like this:

- `dataset.py` decides what the dataset is.
- `encoder.py` decides how text becomes features.
- `data/embed.py` materializes cached artifacts.
- `cbdc/refine.py` and `pipeline/clean.py` build debiased variants.
- `pipeline/classify.py` trains a classifier from cached artifacts.
- `pipeline/evaluate.py` turns saved results into reports.
- `run_all.py` and `run_transformer.py` decide which phases to run and in what order.

## File Map: What Does What

### Core shared files

- `project/config.py`
  - Central config dataclasses and model registry.
  - Change this when you want to add or standardize hyperparameters.

- `project/dataset.py`
  - Loads the dataset and normalizes it into a shared record format.
  - This is where to change column mapping, label mapping, local-file preference, or split behavior.

- `project/encoder.py`
  - Shared encoder abstraction.
  - Handles tokenizer/model loading, hidden-size discovery, transformer-layer resolution, feature pooling, and layer unfreezing.
  - This is the right place to change pooling behavior or model-architecture support.

- `project/losses.py`
  - CBDC-specific loss functions.
  - Refer here when changing the actual debiasing objective.

### Data and caching

- `project/data/embed.py`
  - Phase 1 for both pipelines.
  - Encodes train/val/test and saves:
    - normalized embeddings
    - labels
    - `input_ids`
    - `attention_mask`
    - raw texts and optional metadata

### CBDC path

- `project/cbdc/prompts.py`
  - Prompt bank and prompt encoding utilities for debias_vl/CBDC.
  - Change this when experimenting with confound topics, prompt wording, or mined topic behavior.

- `project/cbdc/refine.py`
  - Phase 2 of the original path.
  - Builds the debias_vl map, discovers CBDC confound directions, trains the CBDC tail, and re-encodes embeddings.

- `project/pipeline/clean.py`
  - Phase 3 of the original path.
  - Applies projection-based cleaning and creates the cleaned experiment conditions.

### Classifier and reporting

- `project/pipeline/classify.py`
  - Phase 4 for both paths.
  - Default mode is the original linear-probe experiment set.
  - Optional mode is transformer fine-tuning using the same cached tokenized data.

- `project/pipeline/evaluate.py`
  - Phase 5 for both paths.
  - Default mode reports original CBDC/linear results.
  - Optional mode reports transformer classifier results.

### Entry scripts

- `project/run_all.py`
  - Main runner for the original CBDC pipeline.

- `project/run_all_debug.py`
  - Debug-oriented version of `run_all.py`.
  - Useful when you want subprocess mode or in-process mode for debugging.

- `project/run_transformer.py`
  - Separate runner for the transformer classifier path.
  - Reuses Phase 1, then runs `classify.py` in transformer mode and `evaluate.py` in transformer mode.

### Cluster helpers

- `project/submit_new.sh`
  - Shell wrapper for running the original `run_all.py` pipeline.

- `project/submit_new.slurm`
  - SLURM job file for running the original `run_all.py` pipeline on a cluster.

Important: the current submit scripts still call `run_all.py`. If you want cluster execution for the transformer path, update the command to run `run_transformer.py`.

## How The Pipelines Run

### Original CBDC path

`python run_all.py`

Phases:

| Phase | Script | Purpose |
|---|---|---|
| 1 | `data/embed.py` | Encode dataset and cache reusable artifacts |
| 2 | `cbdc/refine.py` | Discover confound map, train CBDC tail, re-encode CBDC embeddings |
| 3 | `pipeline/clean.py` | Build projection-cleaned experiment conditions |
| 4 | `pipeline/classify.py` | Train linear probes for all experiment conditions |
| 5 | `pipeline/evaluate.py` | Write the final CBDC-oriented report |

### Transformer path

`python run_transformer.py`

Phases:

| Phase | Script | Purpose |
|---|---|---|
| 1 | `data/embed.py` | Cache embeddings and tokenized splits |
| 2 | `pipeline/classify.py --classifier transformer` | Fine-tune a transformer classifier |
| 3 | `pipeline/evaluate.py --mode transformer` | Write the transformer report |

## Data Contract

The loader expects a 3-way sentiment dataset.

Required columns:

- a text column such as `text`, `tweet`, `sentence`, or `content`
- a label column such as `sentiment`, `label`, or `polarity`

Supported labels are normalized to:

- `0 = negative`
- `1 = neutral`
- `2 = positive`

Optional metadata columns that are kept when present:

- `selected_text`
- `entity`
- `cleaned_text` / `clean_text` / `tokens`
- `Time of Tweet` / `time_of_tweet`
- `Age of User` / `age_of_user`
- `Country` / `country`

By default the loader tries:

1. a local CSV in `project/data/`
2. Kaggle
3. HuggingFace
4. a synthetic fallback

If you already have `project/data/train.csv`, that local file is the main source to care about.

## Cache And Output Layout

When you use the runner scripts, outputs are organized per model:

`project/cache/<model_slug>/`

Example:

- `project/cache/bert/`
- `project/cache/distilbert/`
- `project/cache/finbert/`

Important files produced along the way:

### Phase 1 artifacts

- `z_tweet_train.pt`
- `z_tweet_val.pt`
- `z_tweet_test.pt`

Each contains cached embeddings plus token IDs, attention masks, labels, and texts.

### CBDC path artifacts

- `debias_vl_P.pt`
- `cbdc_directions.pt`
- `sentiment_prototypes.pt`
- `encoder_cbdc.pt`
- `z_tweet_train_cbdc.pt`
- `z_tweet_val_cbdc.pt`
- `z_tweet_test_cbdc.pt`

### Cleaned-condition artifacts

- `z_tweet_*_clean_debias_vl.pt`
- `z_tweet_*_clean_cbdc_directions.pt`
- `z_tweet_*_clean_raw_sentiment_boost.pt`
- `z_tweet_*_clean_cbdc_sentiment_boost.pt`

### Linear-probe outputs

- `probe_<condition>.pt`
- `results.pt`

### Transformer outputs

- `transformer_classifier.pt`
- `transformer_results.pt`

### Reports

- `project/results/eval_report.txt`
- `project/results/transformer_eval_report.txt`

## What You Can Change Safely

These are the safest knobs to change without restructuring the repo.

### Run-time arguments

- backbone model with `--model`
- tokenizer with `--tokenizer`
- max sequence length with `--max_length`
- batch sizes
- number of epochs
- learning rates
- weight decay
- transformer unfreezing depth with `--unfreeze_layers`
- transformer pooling with `--pooling`
- transformer input construction with `--input_mode`
- classifier head choice with `--head_type`
- focal-vs-cross-entropy choice with `--loss_name`
- focal strength with `--focal_gamma`
- whether to train embeddings in transformer mode
- whether to add `Time of Tweet`, `Age of User`, and `Country` to the classifier input
- text-unit wording for prompts with `--text_unit`
- whether to disable sentiment-orthogonal PGD with `--no_sent_orthogonal_pgd`

### Safe code-level extension points

- `dataset.py`
  - new CSV columns
  - new file-discovery rules
  - split policy

- `encoder.py`
  - pooling method
  - support for more HuggingFace architectures
  - trainable-layer strategy

- `cbdc/prompts.py`
  - prompt templates
  - mined topic strategy
  - prompt bank composition

- `pipeline/classify.py`
  - classifier head
  - optimizer policy
  - scheduler policy
  - reporting metrics

- `pipeline/evaluate.py`
  - report layout
  - comparison logic

## What To Change Carefully

These parts are shared contracts between phases. Change them only if you also update downstream consumers.

- cache filenames in each phase
- keys saved inside cached `.pt` payloads
- condition names such as `B1 (raw)` or `D2 (CBDC)`
- the expected meaning of `z_tweet_{split}.pt`
- the meaning of `cbdc_directions.pt` and `sentiment_prototypes.pt`
- the default behavior of `pipeline/classify.py` and `pipeline/evaluate.py`

If your goal is to preserve CBDC correctness, do not casually change:

- the CBDC losses in `losses.py`
- the Phase 2 training loop in `cbdc/refine.py`
- the projection semantics in `pipeline/clean.py`

## When To Refer To Which File

Use this as the fastest navigation guide.

| If you want to... | Start here |
|---|---|
| understand dataset loading | `project/dataset.py` |
| change supported CSV columns | `project/dataset.py` |
| change model/backbone selection | `project/config.py` |
| change tokenization or pooling | `project/encoder.py` |
| inspect what gets cached | `project/data/embed.py` |
| change CBDC prompts | `project/cbdc/prompts.py` |
| change CBDC training behavior | `project/cbdc/refine.py` |
| change projection cleaning | `project/pipeline/clean.py` |
| change classifier training | `project/pipeline/classify.py` |
| change report formatting or comparisons | `project/pipeline/evaluate.py` |
| run the original pipeline | `project/run_all.py` |
| run the transformer pipeline | `project/run_transformer.py` |
| debug runner behavior | `project/run_all_debug.py` |

## Common Workflows

### 1. I want the original method exactly

```bash
cd project
python run_all.py
```

### 2. I want the original structure but without CBDC training

```bash
cd project
python run_all.py --skip_cbdc
```

### 3. I want a plain transformer classifier on the same dataset

```bash
cd project
python run_transformer.py --model distilbert --epochs 3 --unfreeze_layers 2
```

### 4. I want to change prompt design but keep the rest stable

Edit:

- `project/cbdc/prompts.py`

Then rerun at least:

```bash
python run_all.py --start_phase 2
```

### 5. I want to change dataset columns or local file behavior

Edit:

- `project/dataset.py`

Then rerun from Phase 1:

```bash
python run_all.py --start_phase 1
```

or

```bash
python run_transformer.py --start_phase 1
```

### 6. I want to tune only the transformer classifier

Usually the files to inspect are:

- `project/run_transformer.py`
- `project/pipeline/classify.py`
- `project/encoder.py`

Useful knobs for the current `train.csv` setup:

- `--input_mode text_plus_selected`
  - concatenates the tweet text with `selected_text`

- `--input_mode text_selected_pair`
  - encodes the full tweet and `selected_text` separately, then fuses them in the classifier head

- `--use_time_of_tweet`
- `--use_age_of_user`
- `--use_country`
  - inject structured columns into the model input as natural-language prefixes

- `--head_type mlp`
  - uses a stronger MLP head instead of a bare linear layer

- `--loss_name focal`
  - helps the classifier focus more on harder or ambiguous examples

## Environment Variables

The runners mainly communicate with phases through environment variables.

Useful ones:

- `MODEL_NAME`
- `TOKENIZER_NAME`
- `CACHE_DIR`
- `TEXT_UNIT`
- `NO_SENT_ORTHOGONAL_PGD`

If you run phase scripts directly, set these yourself if you want the same behavior as the orchestrators.

## Requirements

Main dependencies are listed in `project/requirements.txt`:

- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `numpy`
- `tqdm`
- `kagglehub[pandas-datasets]`

Install with:

```bash
cd project
python -m pip install -r requirements.txt
```

## Final Notes

- The original CBDC path remains the default path.
- The transformer classifier path is opt-in and separate at the runner level.
- The two paths stay coherent because they share dataset loading, encoder logic, cached splits, classifier/report phases, and model-specific cache directories.

If you are unsure where to begin, start with:

1. `project/run_all.py` if your goal is to study or preserve the CBDC method.
2. `project/run_transformer.py` if your goal is to get a strong direct classifier baseline using the same dataset infrastructure.
