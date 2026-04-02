#!/bin/bash
#SBATCH --job-name=sentiment-sweep
#SBATCH --output=results/slurm_sweep_%j.log
#SBATCH --error=results/slurm_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# 1. Load Conda WITHOUT strict mode (This is what the diagnostic proved works)
source /home/k/kimsh/miniconda3/bin/activate cbdc

# 2. Turn on strict mode ONLY for your pipeline script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_PROJECT_DIR=""
for candidate in "${PROJECT_DIR:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}" "$(pwd)"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}/sweep_submit.sh" ] && [ -f "${candidate}/submit_new.slurm" ]; then
        RESOLVED_PROJECT_DIR="${candidate}"
        break
    fi
done
PROJECT_DIR="${RESOLVED_PROJECT_DIR}"

if [ -z "${PROJECT_DIR}" ]; then
    echo "[ERROR] Could not resolve project directory." >&2
    exit 1
fi

if ! cd "${PROJECT_DIR}"; then
    echo "[ERROR] Unable to cd into project directory: ${PROJECT_DIR}" >&2
    exit 1
fi

mkdir -p "${PROJECT_DIR}/results" "${PROJECT_DIR}/results/sweep"

echo "Job ID:   ${SLURM_JOB_ID:-unknown}"
echo "Node:     $(hostname)"
echo "Time:     $(date)"
echo "Project:  ${PROJECT_DIR}"
echo "Workdir:  $(pwd)"

export PYTHON_BIN="python"

exec bash "${PROJECT_DIR}/sweep_submit.sh" "$@"