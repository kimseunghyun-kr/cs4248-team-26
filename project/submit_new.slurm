#!/bin/bash
#SBATCH --job-name=vlm-personalization
#SBATCH --output=results/slurm_new_%j.log
#SBATCH --error=results/slurm_new_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Personalization launcher for the `vlm-personalization-pgd` branch.

set -euo pipefail
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Time:    $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

ENV_NAME="${ENV_NAME:-cbdc}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -q "Pillow>=10.0.0" 2>/dev/null || true

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

export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=./cache
export HF_HOME=./cache

INSTANCE_DIR="${INSTANCE_DIR:?Set INSTANCE_DIR to the instance image or frame directory}"
CONCEPT_TOKEN="${CONCEPT_TOKEN:?Set CONCEPT_TOKEN, for example 'sks dog'}"
CLASS_NAME="${CLASS_NAME:?Set CLASS_NAME, for example 'a dog'}"

RUN_MODEL="${RUN_MODEL:-qwen25-3b}"
MEDIA_MODE="${MEDIA_MODE:-image}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_DISCOVERY="${SKIP_DISCOVERY:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

RUN_ARGS=(
    --instance_dir "${INSTANCE_DIR}"
    --concept_token "${CONCEPT_TOKEN}"
    --class_name "${CLASS_NAME}"
    --model "${RUN_MODEL}"
    --media_mode "${MEDIA_MODE}"
    --batch_size "${BATCH_SIZE}"
    --max_length "${MAX_LENGTH}"
    --num_workers "${NUM_WORKERS}"
)

if [ -n "${CLASS_DIR:-}" ]; then
    RUN_ARGS+=(--class_dir "${CLASS_DIR}")
fi
if [ -n "${TOKENIZER_NAME:-}" ]; then
    RUN_ARGS+=(--tokenizer "${TOKENIZER_NAME}")
fi
if [ -n "${OUTPUT_DIR}" ]; then
    RUN_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [ "${SKIP_DISCOVERY}" = "1" ]; then
    RUN_ARGS+=(--skip_discovery)
fi

mkdir -p results cache

echo "Running personalization setup with model=${RUN_MODEL} media_mode=${MEDIA_MODE}"
python -u run_personalization.py "${RUN_ARGS[@]}"

echo "Done. $(date)"
