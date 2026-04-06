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

If your environment is already set up, the shortest path is:

```bash
cd project
python run_all.py --classifier transformer --model bert --input_mode text
```

Default behavior runs the transformer classifier pipeline, but for this repo it is usually better to pass `--input_mode text` explicitly.

## Environment Setup

The runtime entry points are:

- local wrapper: `project/submit_new.sh`
- single SLURM worker job: `project/submit_new.slurm`
- sweep wrapper: `project/sweep_submit.slurm`
- sweep generator/analyzer: `project/sweep_submit.sh`

The cluster scripts expect these conda-related environment variables:

- `ENV_NAME`
  default: `cbdc`
- `CONDA_ROOT`
  default: `/home/k/kimsh/miniconda3`
- `CONDA_PROFILE`
  default: `${CONDA_ROOT}/etc/profile.d/conda.sh`

If your conda installation lives somewhere else, you usually do not need to edit the scripts. You can override those values at launch time with `--export=ALL,...`.

### Local or non-SLURM setup

From `project/`:

```bash
python -m pip install -r requirements.txt
python run_all.py --classifier transformer --model bert --input_mode text
```

### SLURM cluster setup with GPU PyTorch

For this repo, the safest cluster flow is:

1. Get an interactive GPU shell.
2. Create or activate the conda env there.
3. Install the CUDA-enabled PyTorch build on that GPU node.
4. Install the remaining Python requirements.
5. Submit jobs through the wrappers.

Example interactive GPU shell matching the current worker script:

```bash
srun --partition=gpu --gres=gpu:a100-40:1 --cpus-per-task=8 --mem=48G --time=02:00:00 --pty bash -l
```

Then, on that GPU shell:

```bash
cd ~/cs4248/project
conda create -n cbdc python=3.11 -y
conda activate cbdc
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
```

Notes:

- The working cluster environment used by this repo reports `torch 2.10.0+cu128`, so the CUDA 12.8 wheel is the intended match.
- `requirements.txt` also includes `torch>=2.0.0`; installing the CUDA wheel first is the important part.
- If your site uses a different GPU resource string or CUDA stack, adjust the `srun` command and PyTorch wheel accordingly.

### Running the local wrapper with conda

`project/submit_new.sh` uses the same conda variables as the SLURM job. Example:

```bash
cd ~/cs4248/project
ENV_NAME=cbdc \
CONDA_ROOT=/home/k/kimsh/miniconda3 \
CONDA_PROFILE=/home/k/kimsh/miniconda3/etc/profile.d/conda.sh \
bash submit_new.sh
```

### Running a single SLURM worker job

From `project/`:

```bash
sbatch \
  --export=ALL,PROJECT_DIR="$PWD",ENV_NAME=cbdc,CONDA_ROOT=/home/k/kimsh/miniconda3,CONDA_PROFILE=/home/k/kimsh/miniconda3/etc/profile.d/conda.sh \
  submit_new.slurm
```

Example overrides:

```bash
sbatch \
  --export=ALL,PROJECT_DIR="$PWD",ENV_NAME=cbdc,CLASSIFIER=linear,MODEL=finbert \
  submit_new.slurm
```

```bash
sbatch \
  --export=ALL,PROJECT_DIR="$PWD",ENV_NAME=cbdc,RUN_NAME=bert_text_u4,CLASSIFIER=transformer,MODEL=bert,INPUT_MODE=text,UNFREEZE_LAYERS=4,HEAD_TYPE=mlp,LOSS_NAME=cross_entropy \
  submit_new.slurm
```

### Running a sweep

The main report sweep:

```bash
cd ~/cs4248/project
sbatch \
  --export=ALL,PROJECT_DIR="$PWD",ENV_NAME=cbdc,CONDA_ROOT=/home/k/kimsh/miniconda3,CONDA_PROFILE=/home/k/kimsh/miniconda3/etc/profile.d/conda.sh \
  sweep_submit.slurm --preset report6h
```

The feature sweep:

```bash
cd ~/cs4248/project
sbatch \
  --export=ALL,PROJECT_DIR="$PWD",ENV_NAME=cbdc,CONDA_ROOT=/home/k/kimsh/miniconda3,CONDA_PROFILE=/home/k/kimsh/miniconda3/etc/profile.d/conda.sh \
  sweep_submit.slurm --preset feature_top5
```

`feature_top5` currently means:

- 5 anchor configs
- 3 feature modes: `vader`, `afinn`, `vader+afinn`
- 2 dataset modes: raw and cleaned
- total: 30 runs

### Changing the SLURM config

To change GPU resources for worker jobs, edit the header in [project/submit_new.slurm](/Users/kimseunghyun/cs4248/cs4248-team-26/project/submit_new.slurm):

- `#SBATCH --partition=gpu`
- `#SBATCH --gres=gpu:a100-40:1`
- `#SBATCH --time=06:00:00`
- `#SBATCH --cpus-per-task=8`
- `#SBATCH --mem=48G`

To change the sweep wrapper placement, edit the header in [project/sweep_submit.slurm](/Users/kimseunghyun/cs4248/cs4248-team-26/project/sweep_submit.slurm):

- `#SBATCH --partition=long`
- `#SBATCH --nodelist=xcnf[0-25]`

That wrapper is pinned to `xcnf` on purpose, because the local Miniconda install is not executable on the ARM `xcng*` nodes.

To change which conda environment the scripts activate, the easiest options are:

- submit with `--export=ALL,ENV_NAME=<your_env>,CONDA_ROOT=<your_conda_root>,CONDA_PROFILE=<your_conda_profile>`
- or edit the defaults near the top of [project/submit_new.slurm](/Users/kimseunghyun/cs4248/cs4248-team-26/project/submit_new.slurm), [project/submit_new.sh](/Users/kimseunghyun/cs4248/cs4248-team-26/project/submit_new.sh), and [project/sweep_submit.slurm](/Users/kimseunghyun/cs4248/cs4248-team-26/project/sweep_submit.slurm)

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
python run_all.py --classifier transformer --model distilbert --input_mode text
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

### Cleaned dataset run

```bash
cd project
python run_all.py --classifier transformer --model bert --input_mode text --use_cleaned_dataset
```

### VADER or VADER+AFINN run

```bash
cd project
python run_all.py --classifier linear --model finbert --use_vader_features
python run_all.py --classifier transformer --model bert --input_mode text --use_vader_features --use_afinn_features
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

If you pass `--use_cleaned_dataset`, it instead looks for cleaned local CSVs such as:

- `project/data/train_cleaned.csv`
- `project/data/test_cleaned.csv`
- accepted test aliases also include `test_clean.csv`

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
sbatch --export=ALL,PROJECT_DIR="$PWD",ENV_NAME=cbdc submit_new.slurm
sbatch --export=ALL,PROJECT_DIR="$PWD",ENV_NAME=cbdc sweep_submit.slurm --preset report6h
```

Useful overrides:

```bash
CLASSIFIER=linear MODEL=finbert bash submit_new.sh
```

```bash
sbatch --export=ALL,PROJECT_DIR="$PWD",ENV_NAME=cbdc,CLASSIFIER=transformer,MODEL=bert,INPUT_MODE=text,UNFREEZE_LAYERS=4 submit_new.slurm
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
