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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_PROJECT_DIR=""
for candidate in "${PROJECT_DIR:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}" "$(pwd)"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}/sweep_submit.sh" ] && [ -f "${candidate}/submit_new.slurm" ]; then
        RESOLVED_PROJECT_DIR="${candidate}"
        break
    fi
done
PROJECT_DIR="${RESOLVED_PROJECT_DIR}"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

if [ -z "${PROJECT_DIR}" ]; then
    echo "[ERROR] Could not resolve project directory." >&2
    echo "[ERROR] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
    echo "[ERROR] SCRIPT_DIR=${SCRIPT_DIR}" >&2
    echo "[ERROR] cwd=$(pwd)" >&2
    exit 1
fi

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

find_conda_sh() {
    if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
        echo "${CONDA_SH}"
        return 0
    fi
    if [ -n "${CONDA_EXE:-}" ]; then
        local conda_base
        conda_base="$(cd "$(dirname "${CONDA_EXE}")/.." 2>/dev/null && pwd || true)"
        if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
            echo "${conda_base}/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    local candidate
    for candidate in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/home/k/kimsh/miniconda3/etc/profile.d/conda.sh" \
        "/home/k/kimsh/anaconda3/etc/profile.d/conda.sh"
    do
        if [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

activate_conda_env() {
    local env_name="$1"
    local conda_sh
    conda_sh="$(find_conda_sh || true)"
    if [ -z "${conda_sh}" ]; then
        echo "[ERROR] Could not locate conda.sh." >&2
        echo "[ERROR] Set CONDA_SH explicitly or submit from an environment with CONDA_EXE set." >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${conda_sh}"
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] 'conda' is unavailable after sourcing ${conda_sh}" >&2
        return 1
    fi
    if ! conda activate "${env_name}"; then
        echo "[ERROR] Failed to activate conda env '${env_name}'" >&2
        return 1
    fi
    return 0
}

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

ENV_NAME="${ENV_NAME:-cbdc}"
activate_conda_env "${ENV_NAME}"
export PYTHON_BIN="$(command -v python)"
echo "Conda env: ${ENV_NAME}"
echo "Python:    ${PYTHON_BIN}"

if ! command -v bash >/dev/null 2>&1; then
    echo "[ERROR] bash is not available in PATH." >&2
    exit 1
fi

exec bash "${PROJECT_DIR}/sweep_submit.sh" "$@"
