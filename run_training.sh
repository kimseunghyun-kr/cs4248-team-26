#!/usr/bin/env bash
# =============================================================================
# run_training.sh  —  Launch model training as a background job
#
# Usage:
#   bash run_training.sh <train_csv> <test_csv> [extra args passed to train_all.py]
#
# Examples:
#   bash run_training.sh data/output_file.csv data/TSAD/test.csv
#   bash run_training.sh data/output_file.csv data/TSAD/test.csv --mlp_epochs 30
#   bash run_training.sh data/output_file.csv data/TSAD/test.csv --text_col text
#
# Outputs:
#   logs/<JOB_ID>.log   — all stdout (progress, metrics, leaderboard)
#   logs/<JOB_ID>.err   — all stderr (warnings, tracebacks)
#   results/result_<JOB_ID>.json  — final summary JSON
# =============================================================================
set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
TRAIN_CSV="${1:?ERROR: train_csv required.  Usage: bash run_training.sh <train_csv> <test_csv>}"
TEST_CSV="${2:?ERROR:  test_csv required.  Usage: bash run_training.sh <train_csv> <test_csv>}"
shift 2   # remaining args forwarded to train_all.py

# ── Job ID & directories ──────────────────────────────────────────────────────
JOB_ID="job_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs"
RESULT_DIR="results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

LOG_FILE="$LOG_DIR/${JOB_ID}.log"
ERR_FILE="$LOG_DIR/${JOB_ID}.err"
RESULT_FILE="$RESULT_DIR/result_${JOB_ID}.json"

# ── Pretty header ─────────────────────────────────────────────────────────────
echo "========================================================"
echo "  Job ID      : $JOB_ID"
echo "  Train CSV   : $TRAIN_CSV"
echo "  Test CSV    : $TEST_CSV"
echo "  Extra args  : $*"
echo "  Log         : $LOG_FILE"
echo "  Stderr      : $ERR_FILE"
echo "  Result JSON : $RESULT_FILE"
echo "========================================================"

# ── Activate conda ────────────────────────────────────────────────────────────
# Works in bash; source the conda init script so 'conda activate' is available.
CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "ERROR: conda not found. Make sure conda is on your PATH." >&2
    exit 1
}
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cs4248

echo "Python  : $(which python)"
echo "Env     : $CONDA_DEFAULT_ENV"
echo ""

# ── Launch training in background ────────────────────────────────────────────
# nohup keeps it running after terminal close.
# stdout → LOG_FILE (also tee'd to terminal so you see live progress)
# stderr → ERR_FILE
nohup python project/train_all.py \
    --train_csv   "$TRAIN_CSV"  \
    --test_csv    "$TEST_CSV"   \
    --job_id      "$JOB_ID"     \
    --output_dir  "$RESULT_DIR" \
    "$@" \
    > >(tee -a "$LOG_FILE") \
    2>"$ERR_FILE" &

BG_PID=$!

echo "Training launched in background."
echo "  PID  : $BG_PID"
echo ""
echo "  Follow live progress :"
echo "    tail -f $LOG_FILE"
echo ""
echo "  Check for errors      :"
echo "    tail -f $ERR_FILE"
echo ""
echo "  Wait for completion   :"
echo "    wait $BG_PID && echo 'DONE'"
echo ""
echo "  Result JSON will be at: $RESULT_FILE"
echo "========================================================"

# Write a small job-info file so you can recover metadata later
cat > "$LOG_DIR/${JOB_ID}.info" <<EOF
job_id     : $JOB_ID
pid        : $BG_PID
train_csv  : $TRAIN_CSV
test_csv   : $TEST_CSV
extra_args : $*
launched   : $(date -u +"%Y-%m-%dT%H:%M:%SZ")
log        : $LOG_FILE
err        : $ERR_FILE
result     : $RESULT_FILE
EOF
