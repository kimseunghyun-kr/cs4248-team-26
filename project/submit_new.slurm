#!/bin/bash
#SBATCH --job-name=cbdc-sentiment
#SBATCH --output=results/slurm_new_%j.log
#SBATCH --error=results/slurm_new_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

# -------------------------------------------------------
# SAE + CBDC Financial Sentiment Pipeline
#
# Usage:
#   sbatch submit_new.slurm
#   sbatch --export=ALL,START_PHASE=4 submit_new.slurm
#   sbatch --export=ALL,ONLY_PHASE=4 submit_new.slurm
#   sbatch --export=ALL,CLASSIFIER=prototype submit_new.slurm
#   sbatch --export=ALL,CLASSIFIER=prototype,INCLUDE_D25=1 submit_new.slurm
#   sbatch --export=ALL,CLASSIFIER=prototype,INCLUDE_D4=1 submit_new.slurm
#   sbatch --export=ALL,RUN_MODEL=bert,CLASSIFIER=prototype,INCLUDE_D4=1 submit_new.slurm
# -------------------------------------------------------

set -euo pipefail
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Time:    $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ---- Activate conda env -------------------------------------------------------
ENV_NAME="cbdc"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ---- Install any new dependencies (idempotent) --------------------------------
pip install -q "kagglehub[pandas-datasets]>=0.3.0" 2>/dev/null || true

# Help large-model loads fragment less on single-GPU jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Packages ready."
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python - <<'PY'
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"torch-visible GPU: {props.name} | total_memory={props.total_memory / 1024**2:.0f} MiB")
PY

# ---- Make output dirs ---------------------------------------------------------
mkdir -p results checkpoints cache

# ---- Run pipeline -------------------------------------------------------------
export PYTHONUNBUFFERED=1

START_PHASE="${START_PHASE:-1}"
ONLY_PHASE="${ONLY_PHASE:-}"
CLASSIFIER="${CLASSIFIER:-probe}"
RUN_MODEL="${RUN_MODEL:-roberta}"
INCLUDE_D25="${INCLUDE_D25:-0}"
INCLUDE_D4="${INCLUDE_D4:-0}"

RUN_ARGS=(--model "${RUN_MODEL}" --classifier "${CLASSIFIER}")
if [ "${INCLUDE_D25}" = "1" ]; then
    RUN_ARGS+=(--include_d25)
fi
if [ "${INCLUDE_D4}" = "1" ]; then
    RUN_ARGS+=(--include_d4)
fi

if [ -n "${ONLY_PHASE}" ]; then
    echo "Running only phase ${ONLY_PHASE} with classifier=${CLASSIFIER} model=${RUN_MODEL} include_d4=${INCLUDE_D4}"
    python -u run_all.py "${RUN_ARGS[@]}" --only_phase "${ONLY_PHASE}"
else
    echo "Running phases from ${START_PHASE} with classifier=${CLASSIFIER} model=${RUN_MODEL} include_d4=${INCLUDE_D4}"
    python -u run_all.py "${RUN_ARGS[@]}" --start_phase "${START_PHASE}"
fi

echo "Done. $(date)"
