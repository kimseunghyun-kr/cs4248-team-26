#!/bin/bash
#SBATCH --job-name=sentiment-sweep
#SBATCH --output=results/slurm_sweep_%j.log
#SBATCH --error=results/slurm_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

on_error() {
    local exit_code="$?"
    local line_no="${1:-unknown}"
    echo "[ERROR] ${SCRIPT_NAME} failed at line ${line_no}: ${BASH_COMMAND}" >&2
    echo "[ERROR] cwd=$(pwd)" >&2
    echo "[ERROR] project_dir=${PROJECT_DIR}" >&2
    echo "[ERROR] slurm_job_id=${SLURM_JOB_ID:-unknown}" >&2
    exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

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

if ! command -v bash >/dev/null 2>&1; then
    echo "[ERROR] bash is not available in PATH." >&2
    exit 1
fi

exec bash "${PROJECT_DIR}/sweep_submit.sh" "$@"
