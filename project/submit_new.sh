#!/bin/bash

# -------------------------------------------------------
# Sentiment Classification Pipeline
#
# Usage:
#   bash submit_new.sh
#   CLASSIFIER=linear bash submit_new.sh
#   START_PHASE=2 bash submit_new.sh
#   ONLY_PHASE=3 bash submit_new.sh
# -------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

on_error() {
    local exit_code="$?"
    local line_no="${1:-unknown}"
    echo "[ERROR] ${SCRIPT_NAME} failed at line ${line_no}: ${BASH_COMMAND}" >&2
    echo "[ERROR] cwd=$(pwd)" >&2
    echo "[ERROR] project_dir=${PROJECT_DIR}" >&2
    exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

if ! cd "${PROJECT_DIR}"; then
    echo "[ERROR] Unable to cd into project directory: ${PROJECT_DIR}" >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Workdir: $(pwd)"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "GPU: N/A"
fi

diag_env() {
    echo "[diag] PATH=${PATH}" >&2
    echo "[diag] which bash=$(command -v bash 2>/dev/null || echo N/A)" >&2
    echo "[diag] which python=$(command -v python 2>/dev/null || echo N/A)" >&2
    echo "[diag] which conda=$(command -v conda 2>/dev/null || echo N/A)" >&2
    echo "[diag] CONDA_ROOT=${CONDA_ROOT:-}" >&2
    echo "[diag] CONDA_PROFILE=${CONDA_PROFILE:-}" >&2
    if command -v file >/dev/null 2>&1; then
        [ -e "${CONDA_ROOT}/bin/python" ] && file "${CONDA_ROOT}/bin/python" >&2 || true
        [ -e "${CONDA_ROOT}/bin/conda" ] && file "${CONDA_ROOT}/bin/conda" >&2 || true
        [ -e "${CONDA_PROFILE}" ] && file "${CONDA_PROFILE}" >&2 || true
    fi
    [ -e "${CONDA_ROOT}/bin/conda" ] && head -1 "${CONDA_ROOT}/bin/conda" >&2 || true
    [ -e "${CONDA_ROOT}/bin/conda" ] && ls -l "${CONDA_ROOT}/bin/conda" >&2 || true
    [ -e "${CONDA_ROOT}/bin/python" ] && ls -l "${CONDA_ROOT}/bin/python" >&2 || true
    [ -e "${CONDA_PROFILE}" ] && ls -l "${CONDA_PROFILE}" >&2 || true
}

probe_conda_local() {
    local probe_rc=0
    echo "[diag] --- local conda probe begin ---" >&2
    if [ -x "${CONDA_ROOT}/bin/python" ]; then
        trap - ERR
        set +e
        "${CONDA_ROOT}/bin/python" -V >&2
        probe_rc=$?
        set -euo pipefail
        trap 'on_error $LINENO' ERR
        echo "[diag] ${CONDA_ROOT}/bin/python -V rc=${probe_rc}" >&2
    else
        echo "[diag] ${CONDA_ROOT}/bin/python is not executable" >&2
    fi

    if [ -x "${CONDA_ROOT}/bin/conda" ]; then
        trap - ERR
        set +e
        "${CONDA_ROOT}/bin/conda" --help >/dev/null
        probe_rc=$?
        set -euo pipefail
        trap 'on_error $LINENO' ERR
        echo "[diag] ${CONDA_ROOT}/bin/conda --help rc=${probe_rc}" >&2

        trap - ERR
        set +e
        "${CONDA_ROOT}/bin/conda" shell.bash hook >/dev/null
        probe_rc=$?
        set -euo pipefail
        trap 'on_error $LINENO' ERR
        echo "[diag] ${CONDA_ROOT}/bin/conda shell.bash hook rc=${probe_rc}" >&2
    else
        echo "[diag] ${CONDA_ROOT}/bin/conda is not executable" >&2
    fi
    echo "[diag] --- local conda probe end ---" >&2
}

ENV_NAME="${ENV_NAME:-cbdc}"
CONDA_ROOT="${CONDA_ROOT:-/home/k/kimsh/miniconda3}"
CONDA_PROFILE="${CONDA_PROFILE:-${CONDA_ROOT}/etc/profile.d/conda.sh}"

echo "[step] local runner: preparing conda env '${ENV_NAME}'"
echo "[step] local runner: CONDA_PROFILE=${CONDA_PROFILE}"

if [ ! -f "${CONDA_PROFILE}" ]; then
    echo "[ERROR] Cannot find conda profile: ${CONDA_PROFILE}" >&2
    diag_env
    probe_conda_local
    exit 1
fi

trap - ERR
set +e +u
# shellcheck disable=SC1090
source "${CONDA_PROFILE}"
source_rc=$?
set -euo pipefail
trap 'on_error $LINENO' ERR

if [ "${source_rc}" -ne 0 ]; then
    echo "[ERROR] Sourcing conda profile failed with rc=${source_rc}" >&2
    diag_env
    probe_conda_local
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 'conda' command unavailable after sourcing ${CONDA_PROFILE}" >&2
    diag_env
    probe_conda_local
    exit 1
fi

trap - ERR
set +e
conda activate "${ENV_NAME}"
activate_rc=$?
set -euo pipefail
trap 'on_error $LINENO' ERR

if [ "${activate_rc}" -ne 0 ]; then
    echo "[ERROR] conda activate '${ENV_NAME}' failed with rc=${activate_rc}" >&2
    diag_env
    probe_conda_local
    exit 1
fi
PYTHON_BIN="python"
echo "Conda env: ${ENV_NAME}"
echo "Python:    $(command -v python)"

"${PYTHON_BIN}" -m pip install -q "kagglehub[pandas-datasets]>=0.3.0" 2>/dev/null || true

echo "Packages ready."

"${PYTHON_BIN}" - <<EOF
import torch
import transformers
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
EOF

export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=./cache
export HF_HOME=./cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "${PROJECT_DIR}/results" "${PROJECT_DIR}/checkpoints" "${PROJECT_DIR}/cache"

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
USE_VADER_FEATURES="${USE_VADER_FEATURES:-0}"
USE_AFINN_FEATURES="${USE_AFINN_FEATURES:-0}"
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
if [ "${USE_VADER_FEATURES}" = "1" ]; then
    CMD+=(--use_vader_features)
fi
if [ "${USE_AFINN_FEATURES}" = "1" ]; then
    CMD+=(--use_afinn_features)
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
