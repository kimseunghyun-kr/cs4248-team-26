#!/bin/bash

# -------------------------------------------------------
# SAE + CBDC Financial Sentiment Pipeline (Standalone)
#
# Usage:
#   bash run_pipeline.sh
#   START_PHASE=4 bash run_pipeline.sh
#   ONLY_PHASE=7 bash run_pipeline.sh
# -------------------------------------------------------

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

# ---- Activate conda env -------------------------------------------------------
ENV_NAME="cbdc"

# load conda properly (important outside slurm)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ---- Install dependencies (idempotent) ---------------------------------------
pip install -q "kagglehub[pandas-datasets]>=0.3.0" 2>/dev/null || true

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

# optional memory tuning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ---- Make output dirs ---------------------------------------------------------
mkdir -p results checkpoints cache

# ---- Run pipeline -------------------------------------------------------------
START_PHASE="${START_PHASE:-1}"
ONLY_PHASE="${ONLY_PHASE:-}"

if [ -n "${ONLY_PHASE}" ]; then
    echo "Running only phase ${ONLY_PHASE}"
    python -u run_all.py --only_phase "${ONLY_PHASE}"
else
    echo "Running phases from ${START_PHASE}"
    python -u run_all.py --start_phase "${START_PHASE}"
fi

echo "Done. $(date)"