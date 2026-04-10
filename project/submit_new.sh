#!/bin/bash

# Personalization launcher for the `vlm-personalization-pgd` branch.

set -euo pipefail

echo "Host: $(hostname)"
echo "Time: $(date)"

# GPU info (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "GPU: N/A"
fi

ENV_NAME="${ENV_NAME:-cbdc}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -q "Pillow>=10.0.0" 2>/dev/null || true

echo "Packages ready."

python - <<EOF
import torch
import transformers
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
EOF

# ---- Env tuning (optional but useful for GPU runs) ----------------------------
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=./cache
export HF_HOME=./cache

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

: "${INSTANCE_DIR:?Set INSTANCE_DIR to the instance image or frame directory}"
: "${CONCEPT_TOKEN:?Set CONCEPT_TOKEN, for example 'sks dog'}"
: "${CLASS_NAME:?Set CLASS_NAME, for example 'a dog'}"

RUN_MODEL="${RUN_MODEL:-qwen25-3b}"
RUN_MODE="${RUN_MODE:-discover}"
MEDIA_MODE="${MEDIA_MODE:-image}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_DISCOVERY="${SKIP_DISCOVERY:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TEXT_ITERS="${TEXT_ITERS:-20}"
IMG_ITERS="${IMG_ITERS:-5}"
TEXT_LR="${TEXT_LR:-1e-3}"
IMG_LR="${IMG_LR:-1e-3}"
IMG_BOUND="${IMG_BOUND:-0.1}"
IMG_STEP="${IMG_STEP:-1e-2}"
IMG_ATTACK_STEPS="${IMG_ATTACK_STEPS:-10}"
IMG_LOSS_SCALE="${IMG_LOSS_SCALE:-100.0}"
IMG_LAMBDA="${IMG_LAMBDA:-1.0}"

RUN_ARGS=(
    --instance_dir "${INSTANCE_DIR}"
    --concept_token "${CONCEPT_TOKEN}"
    --class_name "${CLASS_NAME}"
    --model "${RUN_MODEL}"
    --mode "${RUN_MODE}"
    --media_mode "${MEDIA_MODE}"
    --batch_size "${BATCH_SIZE}"
    --max_length "${MAX_LENGTH}"
    --num_workers "${NUM_WORKERS}"
    --text_iters "${TEXT_ITERS}"
    --img_iters "${IMG_ITERS}"
    --text_lr "${TEXT_LR}"
    --img_lr "${IMG_LR}"
    --img_bound "${IMG_BOUND}"
    --img_step "${IMG_STEP}"
    --img_attack_steps "${IMG_ATTACK_STEPS}"
    --img_loss_scale "${IMG_LOSS_SCALE}"
    --img_lambda "${IMG_LAMBDA}"
)

if [ -n "${CLASS_DIR:-}" ]; then
    RUN_ARGS+=(--class_dir "${CLASS_DIR}")
fi
if [ -n "${TOKENIZER_NAME:-}" ]; then
    RUN_ARGS+=(--tokenizer "${TOKENIZER_NAME}")
fi
if [ -n "${IMAGE_MODEL:-}" ]; then
    RUN_ARGS+=(--image_model "${IMAGE_MODEL}")
fi
if [ -n "${OUTPUT_DIR}" ]; then
    RUN_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [ "${SKIP_DISCOVERY}" = "1" ]; then
    RUN_ARGS+=(--skip_discovery)
fi

mkdir -p results cache

echo "Running personalization setup with model=${RUN_MODEL} mode=${RUN_MODE} media_mode=${MEDIA_MODE}"
python -u run_personalization.py "${RUN_ARGS[@]}"

echo "Done. $(date)"
