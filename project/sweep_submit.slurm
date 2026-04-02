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

diag_env() {
    echo "[diag] PATH=${PATH}" >&2
    echo "[diag] uname -a=$(uname -a 2>/dev/null || echo N/A)" >&2
    echo "[diag] uname -m=$(uname -m 2>/dev/null || echo N/A)" >&2
    echo "[diag] which bash=$(command -v bash 2>/dev/null || echo N/A)" >&2
    echo "[diag] which python=$(command -v python 2>/dev/null || echo N/A)" >&2
    echo "[diag] which python3=$(command -v python3 2>/dev/null || echo N/A)" >&2
    echo "[diag] which conda=$(command -v conda 2>/dev/null || echo N/A)" >&2
    echo "[diag] CONDA_ROOT=${CONDA_ROOT:-}" >&2
    echo "[diag] CONDA_PROFILE=${CONDA_PROFILE:-}" >&2
    if command -v file >/dev/null 2>&1; then
        [ -e "${CONDA_ROOT}/bin/python" ] && file "${CONDA_ROOT}/bin/python" >&2 || true
        [ -e "${CONDA_ROOT}/bin/conda" ] && file "${CONDA_ROOT}/bin/conda" >&2 || true
        [ -e "${CONDA_PROFILE}" ] && file "${CONDA_PROFILE}" >&2 || true
    fi
    [ -e "${CONDA_ROOT}/bin/conda" ] && head -1 "${CONDA_ROOT}/bin/conda" >&2 || true
    [ -e "${CONDA_ROOT}/bin/conda" ] && ls -l "${CONDA_ROOT}/bin/conda" >&2 || true
    [ -e "${CONDA_ROOT}/bin/python" ] && ls -l "${CONDA_ROOT}/bin/python" >&2 || true
    [ -e "${CONDA_PROFILE}" ] && ls -l "${CONDA_PROFILE}" >&2 || true
    [ -e "/usr/bin/python3" ] && file "/usr/bin/python3" >&2 || true
    [ -e "/usr/bin/python3" ] && ls -l "/usr/bin/python3" >&2 || true
}

probe_conda_wrapper() {
    local probe_rc=0
    echo "[diag] --- wrapper conda probe begin ---" >&2
    if [ -x "${CONDA_ROOT}/bin/python" ]; then
        trap - ERR
        set +e
        "${CONDA_ROOT}/bin/python" -V >&2
        probe_rc=$?
        set -euo pipefail
        trap 'on_error $LINENO' ERR
        echo "[diag] ${CONDA_ROOT}/bin/python -V rc=${probe_rc}" >&2
    else
        echo "[diag] ${CONDA_ROOT}/bin/python is not executable" >&2
    fi

    if [ -x "${CONDA_ROOT}/bin/conda" ]; then
        trap - ERR
        set +e
        "${CONDA_ROOT}/bin/conda" --help >/dev/null
        probe_rc=$?
        set -euo pipefail
        trap 'on_error $LINENO' ERR
        echo "[diag] ${CONDA_ROOT}/bin/conda --help rc=${probe_rc}" >&2

        trap - ERR
        set +e
        "${CONDA_ROOT}/bin/conda" shell.bash hook >/dev/null
        probe_rc=$?
        set -euo pipefail
        trap 'on_error $LINENO' ERR
        echo "[diag] ${CONDA_ROOT}/bin/conda shell.bash hook rc=${probe_rc}" >&2
    else
        echo "[diag] ${CONDA_ROOT}/bin/conda is not executable" >&2
    fi
    echo "[diag] --- wrapper conda probe end ---" >&2
}

choose_wrapper_python() {
    local candidate=""
    local resolved=""

    for candidate in /usr/bin/python3 /usr/local/bin/python3 /bin/python3 /usr/bin/python; do
        if [ -x "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done

    resolved="$(PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin command -v python3 2>/dev/null || true)"
    if [ -n "${resolved}" ] && [[ "${resolved}" != "${CONDA_ROOT}"* ]]; then
        echo "${resolved}"
        return 0
    fi

    resolved="$(PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin command -v python 2>/dev/null || true)"
    if [ -n "${resolved}" ] && [[ "${resolved}" != "${CONDA_ROOT}"* ]]; then
        echo "${resolved}"
        return 0
    fi

    return 1
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
echo "Args:     $*"

ENV_NAME="${ENV_NAME:-cbdc}"
CONDA_ROOT="${CONDA_ROOT:-/home/k/kimsh/miniconda3}"
CONDA_PROFILE="${CONDA_PROFILE:-${CONDA_ROOT}/etc/profile.d/conda.sh}"

echo "[step] sweep wrapper: preparing conda env '${ENV_NAME}'"
echo "[step] sweep wrapper: CONDA_PROFILE=${CONDA_PROFILE}"

if [ -f "${CONDA_PROFILE}" ]; then
    trap - ERR
    set +e +u
    # shellcheck disable=SC1090
    source "${CONDA_PROFILE}"
    source_rc=$?
    set -euo pipefail
    trap 'on_error $LINENO' ERR
else
    source_rc=1
fi

activate_rc=1
if [ "${source_rc}" -eq 0 ] && command -v conda >/dev/null 2>&1; then
    echo "[step] sweep wrapper: sourced conda profile successfully"
    trap - ERR
    set +e
    conda activate "${ENV_NAME}"
    activate_rc=$?
    set -euo pipefail
    trap 'on_error $LINENO' ERR
fi

if [ "${source_rc}" -eq 0 ] && [ "${activate_rc}" -eq 0 ]; then
    export PYTHON_BIN="python"
    echo "Conda env: ${ENV_NAME}"
    echo "Python:    $(command -v python)"
else
    echo "[warn] sweep wrapper could not activate conda env '${ENV_NAME}'" >&2
    echo "[warn] source_rc=${source_rc} activate_rc=${activate_rc}" >&2
    echo "[warn] wrapper will fall back to system python for matrix generation/log parsing only" >&2
    diag_env
    probe_conda_wrapper
    if ! PYTHON_BIN="$(choose_wrapper_python)"; then
        echo "[ERROR] No fallback python interpreter available for sweep wrapper." >&2
        exit 1
    fi
    export PYTHON_BIN
    echo "Conda env: unavailable for wrapper"
    echo "Python:    ${PYTHON_BIN}"
    "${PYTHON_BIN}" -V || true
fi

if ! command -v bash >/dev/null 2>&1; then
    echo "[ERROR] bash is not available in PATH." >&2
    diag_env
    exit 1
fi

echo "[step] sweep wrapper: executing sweep_submit.sh $*"
exec bash "${PROJECT_DIR}/sweep_submit.sh" "$@"
