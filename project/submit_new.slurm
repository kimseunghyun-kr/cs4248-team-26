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
#
# Usage:
#   sbatch submit_new.slurm
#   sbatch --export=ALL,CLASSIFIER=linear submit_new.slurm
#   sbatch --export=ALL,START_PHASE=2 submit_new.slurm
#   sbatch --export=ALL,ONLY_PHASE=3 submit_new.slurm
# -------------------------------------------------------

set -euo pipefail

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Time:    $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

ENV_NAME="${ENV_NAME:-sentiment}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -q "kagglehub[pandas-datasets]>=0.3.0" 2>/dev/null || true

echo "Packages ready."
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

mkdir -p results checkpoints cache
export PYTHONUNBUFFERED=1

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
