# Sentiment Classification Pipeline

This repository is now centered on one reusable 3-way sentiment classification pipeline.

It supports two classifier modes on the same dataset/cache structure:

- `linear`: a linear probe on cached transformer embeddings
- `transformer`: end-to-end transformer fine-tuning for stronger results

The old CBDC / DebiasVL path has been removed from the runtime pipeline.

## Pipeline Shape

The core flow is:

`dataset.py -> data/embed.py -> pipeline/classify.py -> pipeline/evaluate.py`

The runner is:

`project/run_all.py`

## Quick Start

From `project/`:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Default behavior runs the transformer classifier pipeline.

Backbone shortcuts now include both encoder-style and decoder-style models:

- `bert`
- `finbert`
- `bertweet`
- `roberta`
- `distilbert`
- `mistral`
- `tinyllama`
- `qwen2`

You can also pass any Hugging Face model ID directly with `--model`.

## What To Run

### Recommended transformer run

```bash
cd project
python run_all.py \
  --classifier transformer \
  --model distilbert \
  --epochs 3 \
  --unfreeze_layers 2 \
  --input_mode text_selected_pair \
  --head_type mlp \
  --loss_name focal \
  --use_time_of_tweet \
  --use_age_of_user
```

### Simple transformer baseline

```bash
cd project
python run_all.py --classifier transformer --model distilbert
```

### Decoder-style backbone examples

```bash
cd project
python run_all.py --classifier transformer --model tinyllama
python run_all.py --classifier transformer --model qwen2
python run_all.py --classifier transformer --model mistral --pooling auto
```

### Linear probe baseline

```bash
cd project
python run_all.py --classifier linear --model distilbert
```

### Resume or run a single phase

```bash
python run_all.py --start_phase 2
python run_all.py --only_phase 2
python run_all.py --classifier linear --only_phase 3
```

## Phases

### Phase 1: `project/data/embed.py`

Purpose:
- loads the dataset
- tokenizes text
- caches embeddings, labels, token ids, masks, and optional metadata

Outputs in `project/cache/<model_slug>/`:
- `z_tweet_train.pt`
- `z_tweet_val.pt`
- `z_tweet_test.pt`

### Phase 2: `project/pipeline/classify.py`

Purpose:
- `linear` mode trains a linear probe on cached embeddings
- `transformer` mode fine-tunes the encoder plus classifier head

Outputs:
- linear: `linear_results.pt`, `linear_probe.pt`
- transformer: `transformer_results.pt`, `transformer_classifier.pt`

### Phase 3: `project/pipeline/evaluate.py`

Purpose:
- loads the saved results from Phase 2
- writes a report into `project/results/`

Outputs:
- linear: `linear_eval_report.txt`
- transformer: `transformer_eval_report.txt`

## File Map

### Core files

- `project/config.py`
  - model registry and transformer classifier config
  - change this when you want to standardize hyperparameters

- `project/dataset.py`
  - dataset loading, column normalization, split logic, PyTorch dataset helpers
  - change this when dataset columns or text construction need to change

- `project/encoder.py`
  - shared encoder abstraction
  - handles tokenizer/model loading, pooling, and selective unfreezing
  - `auto` pooling uses CLS for encoder-style models and last-token pooling for decoder-style models
  - change this when adjusting backbone behavior

- `project/data/embed.py`
  - caches embeddings and tokens for train/val/test
  - change this when cached payload contents should change

- `project/pipeline/classify.py`
  - owns model training for both classifier modes
  - change this when adding losses, heads, or training behavior

- `project/pipeline/evaluate.py`
  - owns report generation
  - change this when report contents or comparison logic should change

- `project/run_all.py`
  - single entry point for the whole pipeline
  - change this when phase orchestration or CLI behavior should change

### Helpers

- `project/submit_new.sh`
  - shell wrapper for local or server runs

- `project/submit_new.slurm`
  - SLURM wrapper for cluster runs

## Dataset Contract

The loader expects a 3-way sentiment dataset.

Required columns:
- a text column such as `text`, `tweet`, `sentence`, or `content`
- a label column such as `sentiment`, `label`, or `polarity`

Normalized labels:
- `0 = negative`
- `1 = neutral`
- `2 = positive`

Optional columns kept when present:
- `selected_text`
- `Time of Tweet`
- `Age of User`
- `Country`
- `entity`
- token/cleaned-text fields

The local dataset loader prefers CSV files in `project/data/`, including your current `project/data/train.csv`.

## Transformer Methods Available

The transformer path adds a few stronger options for the 3-way classifier.

### Input construction

- `--input_mode text`
  - just the main text

- `--input_mode text_plus_selected`
  - appends `selected_text` into one sequence

- `--input_mode text_selected_pair`
  - encodes main text and `selected_text` separately, then fuses both views

### Metadata prefixes

Optional flags:
- `--use_time_of_tweet`
- `--use_age_of_user`
- `--use_country`

These prepend lightweight structured context to the text view.

### Classifier head

- `--head_type linear`
- `--head_type mlp`

### Loss

- `--loss_name cross_entropy`
- `--loss_name focal`

## What To Change For Common Tasks

### Change dataset handling

Edit:
- `project/dataset.py`

Use this for:
- column mapping
- local CSV selection
- split behavior
- text-view construction

### Change the model backbone or pooling

Edit:
- `project/config.py`
- `project/encoder.py`

Use this for:
- adding new Hugging Face backbone shortcuts
- changing pooling behavior
- changing which transformer layers unfreeze

### Change training behavior

Edit:
- `project/pipeline/classify.py`

Use this for:
- learning rates
- new classifier heads
- new losses
- class weighting
- early stopping

### Change what gets reported

Edit:
- `project/pipeline/evaluate.py`

## Cluster Usage

From `project/`:

```bash
bash submit_new.sh
sbatch submit_new.slurm
```

Useful overrides:

```bash
CLASSIFIER=linear bash submit_new.sh
MODEL=distilbert EPOCHS=3 UNFREEZE_LAYERS=2 bash submit_new.sh
sbatch --export=ALL,CLASSIFIER=transformer,MODEL=distilbert,EPOCHS=3 submit_new.slurm
```

## Suggested Experiments

### 1. Linear baseline

```bash
python run_all.py --classifier linear --model distilbert
```

### 2. Plain transformer baseline

```bash
python run_all.py --classifier transformer --model distilbert --input_mode text
```

### 3. Add selected-text signal

```bash
python run_all.py --classifier transformer --model distilbert --input_mode text_plus_selected
```

### 4. Stronger paired-view transformer

```bash
python run_all.py \
  --classifier transformer \
  --model distilbert \
  --input_mode text_selected_pair \
  --head_type mlp \
  --loss_name focal
```

### 5. Add metadata

```bash
python run_all.py \
  --classifier transformer \
  --model distilbert \
  --input_mode text_selected_pair \
  --head_type mlp \
  --loss_name focal \
  --use_time_of_tweet \
  --use_age_of_user
```

## Current Recommendation

If the goal is the best practical classifier for `project/data/train.csv`, start with:

```bash
python run_all.py \
  --classifier transformer \
  --model distilbert \
  --epochs 3 \
  --unfreeze_layers 2 \
  --input_mode text_selected_pair \
  --head_type mlp \
  --loss_name focal \
  --use_time_of_tweet \
  --use_age_of_user
```
