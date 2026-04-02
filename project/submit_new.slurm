#!/bin/bash
#SBATCH --job-name=cbdc-sentiment
#SBATCH --output=results/slurm_new_%j.log
#SBATCH --error=results/slurm_new_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# -------------------------------------------------------
# SAE + CBDC Financial Sentiment Pipeline
#
# Usage:
#   sbatch submit_new.slurm                       # run all phases (1-7)
#   sbatch --export=ALL,START_PHASE=4 submit_new.slurm  # resume from phase 4
#   sbatch --export=ALL,ONLY_PHASE=7  submit_new.slurm  # run only eval
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

echo "Packages ready."
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

# ---- Make output dirs ---------------------------------------------------------
mkdir -p results checkpoints cache

# ---- Run pipeline -------------------------------------------------------------
export PYTHONUNBUFFERED=1

START_PHASE="${START_PHASE:-1}"
ONLY_PHASE="${ONLY_PHASE:-}"

if [ -n "${ONLY_PHASE}" ]; then
    echo "Running only phase ${ONLY_PHASE}"
    python -u run_all.py --only_phase "${ONLY_PHASE}"
else
    echo "Running phases from ${START_PHASE}"
    python -u run_all.py --model "roberta" --start_phase "${START_PHASE}"
fi

echo "Done. $(date)"
