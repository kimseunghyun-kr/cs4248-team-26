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

echo "Host: $(hostname)"
echo "Time: $(date)"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "GPU: N/A"
fi

ENV_NAME="${ENV_NAME:-sentiment}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -q "kagglehub[pandas-datasets]>=0.3.0" 2>/dev/null || true

echo "Packages ready."

python - <<EOF
import torch
import transformers
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
EOF

export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=./cache
export HF_HOME=./cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p results checkpoints cache

CLASSIFIER="${CLASSIFIER:-transformer}"
MODEL="${MODEL:-bert}"
START_PHASE="${START_PHASE:-1}"
ONLY_PHASE="${ONLY_PHASE:-}"
MAX_LENGTH="${MAX_LENGTH:-128}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-4}"
INPUT_MODE="${INPUT_MODE:-text_plus_selected}"
HEAD_TYPE="${HEAD_TYPE:-mlp}"
LOSS_NAME="${LOSS_NAME:-cross_entropy}"

CMD=(
  python -u run_all.py
  --classifier "${CLASSIFIER}"
  --model "${MODEL}"
  --max_length "${MAX_LENGTH}"
  --embed_batch_size "${EMBED_BATCH_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
)

if [ "${CLASSIFIER}" = "transformer" ]; then
  CMD+=(
    --epochs "${EPOCHS}"
    --unfreeze_layers "${UNFREEZE_LAYERS}"
    --input_mode "${INPUT_MODE}"
    --head_type "${HEAD_TYPE}"
    --loss_name "${LOSS_NAME}"
  )
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
