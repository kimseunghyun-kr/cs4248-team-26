#!/bin/bash
#SBATCH --job-name=sentiment-clf
#SBATCH --output=results/slurm_new_%j.log
#SBATCH --error=results/slurm_new_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# -------------------------------------------------------
# Sentiment Classification Pipeline
# -------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_PROJECT_DIR=""
for candidate in "${PROJECT_DIR:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}" "$(pwd)"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}/run_all.py" ] && [ -f "${candidate}/submit_new.slurm" ]; then
        RESOLVED_PROJECT_DIR="${candidate}"
        break
    fi
done
PROJECT_DIR="${RESOLVED_PROJECT_DIR}"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

if [ -z "${PROJECT_DIR}" ]; then
    echo "[ERROR] Could not resolve project directory." >&2
    echo "[ERROR] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
    echo "[ERROR] SCRIPT_DIR=${SCRIPT_DIR}" >&2
    echo "[ERROR] cwd=$(pwd)" >&2
    exit 1
fi

on_error() {
    local exit_code="$?"
    local line_no="${1:-unknown}"
    echo "[ERROR] ${SCRIPT_NAME} failed at line ${line_no}: ${BASH_COMMAND}" >&2
    echo "[ERROR] cwd=$(pwd)" >&2
    echo "[ERROR] project_dir=${PROJECT_DIR}" >&2
    echo "[ERROR] slurm_job_id=${SLURM_JOB_ID:-unknown}" >&2
    exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

if ! cd "${PROJECT_DIR}"; then
    echo "[ERROR] Unable to cd into project directory: ${PROJECT_DIR}" >&2
    exit 1
fi

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Time:    $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Project: ${PROJECT_DIR}"
echo "Workdir: $(pwd)"

# ========================================================
# DIRECT BASH ACTIVATION (BYPASSES PYTHON CONDA WRAPPER)
# ========================================================
ENV_NAME="${ENV_NAME:-cbdc}"

if [ -f "/home/k/kimsh/miniconda3/bin/activate" ]; then
    source "/home/k/kimsh/miniconda3/bin/activate" "${ENV_NAME}"
else
    echo "[ERROR] Cannot find raw activate script at /home/k/kimsh/miniconda3/bin/activate" >&2
    exit 1
fi

export PYTHON_BIN="python"
echo "Conda env: ${ENV_NAME}"
echo "Python:    $(command -v python)"
# ========================================================

"${PYTHON_BIN}" -m pip install -q "kagglehub[pandas-datasets]>=0.3.0" 2>/dev/null || true

echo "Packages ready."
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
"${PYTHON_BIN}" -c "import transformers; print('transformers', transformers.__version__)"

mkdir -p "${PROJECT_DIR}/results" "${PROJECT_DIR}/checkpoints" "${PROJECT_DIR}/cache"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-}"
CLASSIFIER="${CLASSIFIER:-transformer}"
MODEL="${MODEL:-bert}"
TOKENIZER="${TOKENIZER:-}"
START_PHASE="${START_PHASE:-1}"
ONLY_PHASE="${ONLY_PHASE:-}"
MAX_LENGTH="${MAX_LENGTH:-128}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
ENCODER_LR="${ENCODER_LR:-2e-5}"
CLASSIFIER_LR="${CLASSIFIER_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
POOLING="${POOLING:-auto}"
DROPOUT="${DROPOUT:-0.1}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-4}"
INPUT_MODE="${INPUT_MODE:-text_plus_selected}"
HEAD_TYPE="${HEAD_TYPE:-mlp}"
HIDDEN_DIM="${HIDDEN_DIM:-0}"
LOSS_NAME="${LOSS_NAME:-cross_entropy}"
FOCAL_GAMMA="${FOCAL_GAMMA:-1.5}"
PATIENCE="${PATIENCE:-2}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
TRAIN_EMBEDDINGS="${TRAIN_EMBEDDINGS:-0}"
USE_TIME_OF_TWEET="${USE_TIME_OF_TWEET:-0}"
USE_AGE_OF_USER="${USE_AGE_OF_USER:-0}"
USE_COUNTRY="${USE_COUNTRY:-0}"
NO_CLASS_WEIGHTS="${NO_CLASS_WEIGHTS:-0}"

if [ -n "${RUN_NAME}" ]; then
    echo "Run name: ${RUN_NAME}"
fi

CMD=(
  "${PYTHON_BIN}" -u run_all.py
  --classifier "${CLASSIFIER}"
  --model "${MODEL}"
  --max_length "${MAX_LENGTH}"
  --embed_batch_size "${EMBED_BATCH_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --encoder_lr "${ENCODER_LR}"
  --classifier_lr "${CLASSIFIER_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --pooling "${POOLING}"
  --dropout "${DROPOUT}"
)

if [ -n "${RUN_NAME}" ]; then
    CMD+=(--run_name "${RUN_NAME}")
fi

if [ "${CLASSIFIER}" = "transformer" ]; then
  CMD+=(
    --epochs "${EPOCHS}"
    --unfreeze_layers "${UNFREEZE_LAYERS}"
    --input_mode "${INPUT_MODE}"
    --head_type "${HEAD_TYPE}"
    --hidden_dim "${HIDDEN_DIM}"
    --loss_name "${LOSS_NAME}"
    --focal_gamma "${FOCAL_GAMMA}"
    --patience "${PATIENCE}"
    --label_smoothing "${LABEL_SMOOTHING}"
    --grad_clip_norm "${GRAD_CLIP_NORM}"
  )
fi

if [ -n "${TOKENIZER}" ]; then
    CMD+=(--tokenizer "${TOKENIZER}")
fi

if [ "${TRAIN_EMBEDDINGS}" = "1" ]; then
    CMD+=(--train_embeddings)
fi
if [ "${USE_TIME_OF_TWEET}" = "1" ]; then
    CMD+=(--use_time_of_tweet)
fi
if [ "${USE_AGE_OF_USER}" = "1" ]; then
    CMD+=(--use_age_of_user)
fi
if [ "${USE_COUNTRY}" = "1" ]; then
    CMD+=(--use_country)
fi
if [ "${NO_CLASS_WEIGHTS}" = "1" ]; then
    CMD+=(--no_class_weights)
fi

if [ -n "${ONLY_PHASE}" ]; then
    echo "Running only phase ${ONLY_PHASE}"
    CMD+=(--only_phase "${ONLY_PHASE}")
else
    echo "Running phases from ${START_PHASE}"
    CMD+=(--start_phase "${START_PHASE}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Done. $(date)"