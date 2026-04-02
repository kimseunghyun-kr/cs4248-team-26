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

activate_conda_for_sweep() {
    local env_name="$1"
    local candidate

    if [ -f "${HOME}/.bashrc" ]; then
        trap - ERR
        set +e +u
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
        local bashrc_rc=$?
        set -euo pipefail
        trap 'on_error $LINENO' ERR
        if [ "${bashrc_rc}" -eq 0 ] && command -v conda >/dev/null 2>&1 && conda activate "${env_name}" >/dev/null 2>&1; then
            return 0
        fi
        echo "[warn] ~/.bashrc did not fully initialize conda; trying conda.sh fallback" >&2
    fi

    for candidate in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/home/k/kimsh/miniconda3/etc/profile.d/conda.sh" \
        "/home/k/kimsh/anaconda3/etc/profile.d/conda.sh"
    do
        if [ ! -f "${candidate}" ]; then
            continue
        fi
        trap - ERR
        set +e +u
        # shellcheck disable=SC1090
        source "${candidate}"
        local conda_sh_rc=$?
        set -euo pipefail
        trap 'on_error $LINENO' ERR
        if [ "${conda_sh_rc}" -ne 0 ]; then
            continue
        fi
        if command -v conda >/dev/null 2>&1 && conda activate "${env_name}" >/dev/null 2>&1; then
            return 0
        fi
    done

    return 1
}

ENV_NAME="${ENV_NAME:-cbdc}"
if ! activate_conda_for_sweep "${ENV_NAME}"; then
    echo "[ERROR] Failed to activate conda env '${ENV_NAME}' for sweep wrapper." >&2
    exit 1
fi
export PYTHON_BIN="python"
echo "Conda env: ${ENV_NAME}"
echo "Python:    $(command -v python)"

if ! command -v bash >/dev/null 2>&1; then
    echo "[ERROR] bash is not available in PATH." >&2
    exit 1
fi

exec bash "${PROJECT_DIR}/sweep_submit.sh" "$@"
