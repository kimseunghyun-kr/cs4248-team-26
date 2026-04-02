#!/bin/bash
#SBATCH --job-name=sentiment-sweep
#SBATCH --output=results/slurm_sweep_%j.log
#SBATCH --error=results/slurm_sweep_%j.err
#SBATCH --time=12:00:00
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

print_sweep_diagnostics() {
    echo "[diag] uname: $(uname -a)" >&2
    echo "[diag] PATH: ${PATH}" >&2
    echo "[diag] which python: $(command -v python 2>/dev/null || echo N/A)" >&2
    echo "[diag] which python3: $(command -v python3 2>/dev/null || echo N/A)" >&2
    echo "[diag] which conda: $(command -v conda 2>/dev/null || echo N/A)" >&2
    if command -v file >/dev/null 2>&1; then
        [ -e "/home/k/kimsh/miniconda3/bin/python" ] && file /home/k/kimsh/miniconda3/bin/python >&2 || true
        [ -e "/home/k/kimsh/miniconda3/bin/conda" ] && file /home/k/kimsh/miniconda3/bin/conda >&2 || true
    fi
    [ -e "/home/k/kimsh/miniconda3/bin/conda" ] && head -1 /home/k/kimsh/miniconda3/bin/conda >&2 || true
}

try_activate_sweep_conda() {
    local env_name="$1"
    local conda_profile="/home/k/kimsh/miniconda3/etc/profile.d/conda.sh"
    local source_rc=1
    local activate_rc=1

    trap - ERR
    set +e +u
    if [ -f "${conda_profile}" ]; then
        # shellcheck disable=SC1090
        source "${conda_profile}"
        source_rc=$?
        if [ "${source_rc}" -eq 0 ] && command -v conda >/dev/null 2>&1; then
            conda activate "${env_name}"
            activate_rc=$?
        fi
    fi
    set -euo pipefail
    trap 'on_error $LINENO' ERR

    [ "${source_rc}" -eq 0 ] && [ "${activate_rc}" -eq 0 ]
}

ENV_NAME="${ENV_NAME:-cbdc}"
if try_activate_sweep_conda "${ENV_NAME}"; then
    export PYTHON_BIN="python"
    echo "Conda env: ${ENV_NAME}"
    echo "Python:    $(command -v python)"
else
    echo "[warn] Sweep wrapper could not activate conda env '${ENV_NAME}' on node $(hostname)." >&2
    print_sweep_diagnostics
    if command -v python3 >/dev/null 2>&1; then
        export PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        export PYTHON_BIN="python"
    else
        echo "[ERROR] Sweep wrapper could not find python3 or python after conda activation failed." >&2
        exit 1
    fi
    echo "Conda env: unavailable for wrapper; using ${PYTHON_BIN}"
    echo "Python:    $(command -v "${PYTHON_BIN}")"
fi

if ! command -v bash >/dev/null 2>&1; then
    echo "[ERROR] bash is not available in PATH." >&2
    exit 1
fi

exec bash "${PROJECT_DIR}/sweep_submit.sh" "$@"
