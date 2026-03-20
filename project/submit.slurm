#SBATCH --job-name=cbdc-sentiment
#SBATCH --output=results/slurm_%j.log
#SBATCH --error=results/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1       # MUST have this to use xgpg nodes
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# -------------------------------------------------------
# Usage:
#   sbatch submit.slurm
#
# To override defaults:
#   sbatch --time=02:00:00 submit.slurm
# -------------------------------------------------------

set -euo pipefail
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Time:   $(date)"

# ---- Load modules (adjust for your cluster) -----------
# module load cuda/12.1
# module load anaconda3

# ---- Activate conda env directly (avoids conda run buffering) --------
ENV_NAME="cbdc"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Packages ready."
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

# ---- Make results dir ---------------------------------
mkdir -p results checkpoints

# ---- Run pipeline -------------------------------------
export PYTHONUNBUFFERED=1
python -u run_pipeline.py

echo "Done. $(date)"
