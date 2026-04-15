#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  Train all models, then tune the best ones (single job)
#
# Usage:
#   bash run_pipeline.sh <train_csv> <test_csv> [options]
#
# Options:
#   --n_trials N        Optuna trials per model family   (default 50)
#   --text_col COL      Column name for text             (default cleaned_text)
#   --val_size F        Validation fraction              (default 0.1)
#   --random_state S    Global seed                      (default 42)
#   --mlp_epochs N      MLP training epochs              (default 20)
#   --mlp_batch_size N  MLP batch size                   (default 256)
#   --mlp_lr F          MLP learning rate                (default 0.001)
#   --mlp_hidden_dim N  MLP hidden layer width           (default 512)
#
# Example:
#   bash run_pipeline.sh data/output_file.csv data/TSAD/test.csv --n_trials 80
#
# Outputs (all under the same JOB_ID):
#   logs/<JOB_ID>_train.log   — training stdout
#   logs/<JOB_ID>_tune.log    — tuning stdout
#   logs/<JOB_ID>.err         — all stderr (both phases)
#   logs/<JOB_ID>.info        — metadata (PID, paths, launch time)
#   results/result_<JOB_ID>.json  — training summary
#   results/tuned_<JOB_ID>.json   — tuning summary
# =============================================================================
set -euo pipefail

# ── Required positional args ──────────────────────────────────────────────────
TRAIN_CSV="${1:?Usage: bash run_pipeline.sh <train_csv> <test_csv> [options]}"
TEST_CSV="${2:?Usage: bash run_pipeline.sh <train_csv> <test_csv> [options]}"
shift 2

# ── Optional keyword args (with defaults) ────────────────────────────────────
N_TRIALS=50
TEXT_COL="cleaned_text"
VAL_SIZE=0.1
RANDOM_STATE=42
MLP_EPOCHS=20
MLP_BATCH_SIZE=256
MLP_LR=0.001
MLP_HIDDEN_DIM=512

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_trials)       N_TRIALS="$2";       shift 2 ;;
        --text_col)       TEXT_COL="$2";       shift 2 ;;
        --val_size)       VAL_SIZE="$2";       shift 2 ;;
        --random_state)   RANDOM_STATE="$2";   shift 2 ;;
        --mlp_epochs)     MLP_EPOCHS="$2";     shift 2 ;;
        --mlp_batch_size) MLP_BATCH_SIZE="$2"; shift 2 ;;
        --mlp_lr)         MLP_LR="$2";         shift 2 ;;
        --mlp_hidden_dim) MLP_HIDDEN_DIM="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Paths ────────────────────────────────────────────────────────────────────
JOB_ID="job_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs"
RESULT_DIR="results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

TRAIN_LOG="$LOG_DIR/${JOB_ID}_train.log"
TUNE_LOG="$LOG_DIR/${JOB_ID}_tune.log"
ERR_FILE="$LOG_DIR/${JOB_ID}.err"
TRAIN_RESULT="$RESULT_DIR/result_${JOB_ID}.json"
TUNE_RESULT="$RESULT_DIR/tuned_${JOB_ID}.json"

# ── Header ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  Job ID        : $JOB_ID"
echo "  Train CSV     : $TRAIN_CSV"
echo "  Test CSV      : $TEST_CSV"
echo "  Optuna trials : $N_TRIALS"
echo "  MLP epochs    : $MLP_EPOCHS"
echo "  Train log     : $TRAIN_LOG"
echo "  Tune log      : $TUNE_LOG"
echo "  Errors        : $ERR_FILE"
echo "  Train result  : $TRAIN_RESULT"
echo "  Tune result   : $TUNE_RESULT"
echo "========================================================"

# ── Activate conda ───────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "ERROR: conda not found on PATH." >&2; exit 1
}
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cs4248

echo "Python : $(which python)"
echo "Env    : $CONDA_DEFAULT_ENV"
echo ""

# ── Write runner script (values baked in at write-time) ───────────────────────
RUNNER="/tmp/pipeline_${JOB_ID}.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -eo pipefail

# Re-activate conda inside the nohup sub-shell
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cs4248

echo "[\$(date)] ===== PHASE 1: TRAINING =====" >> "${ERR_FILE}"

python project/train_all.py \\
    --train_csv    "${TRAIN_CSV}"     \\
    --test_csv     "${TEST_CSV}"      \\
    --job_id       "${JOB_ID}"        \\
    --output_dir   "${RESULT_DIR}"    \\
    --text_col     "${TEXT_COL}"      \\
    --val_size     "${VAL_SIZE}"      \\
    --random_state "${RANDOM_STATE}"  \\
    --mlp_epochs   "${MLP_EPOCHS}"    \\
    --mlp_batch_size "${MLP_BATCH_SIZE}" \\
    --mlp_lr       "${MLP_LR}"        \\
    --mlp_hidden_dim "${MLP_HIDDEN_DIM}" \\
    >> "${TRAIN_LOG}" 2>> "${ERR_FILE}"

if [ \$? -ne 0 ]; then
    echo "[\$(date)] train_all.py FAILED — aborting pipeline" >> "${ERR_FILE}"
    exit 1
fi
echo "[\$(date)] Training complete -> ${TRAIN_RESULT}" >> "${ERR_FILE}"

echo "[\$(date)] ===== PHASE 2: TUNING =====" >> "${ERR_FILE}"

python project/tune_all.py \\
    --train_results "${TRAIN_RESULT}" \\
    --train_csv    "${TRAIN_CSV}"     \\
    --test_csv     "${TEST_CSV}"      \\
    --job_id       "${JOB_ID}"        \\
    --output_dir   "${RESULT_DIR}"    \\
    --text_col     "${TEXT_COL}"      \\
    --val_size     "${VAL_SIZE}"      \\
    --random_state "${RANDOM_STATE}"  \\
    --n_trials     "${N_TRIALS}"      \\
    --mlp_epochs   "${MLP_EPOCHS}"    \\
    >> "${TUNE_LOG}" 2>> "${ERR_FILE}"

if [ \$? -ne 0 ]; then
    echo "[\$(date)] tune_all.py FAILED" >> "${ERR_FILE}"
    exit 1
fi
echo "[\$(date)] Tuning complete -> ${TUNE_RESULT}" >> "${ERR_FILE}"
echo "[\$(date)] ===== PIPELINE DONE =====" >> "${ERR_FILE}"
EOF

chmod +x "$RUNNER"

# ── Launch in background (survives terminal close via nohup) ──────────────────
nohup bash "$RUNNER" &
BG_PID=$!

echo "Pipeline running in background (PID=$BG_PID)"
echo ""
echo "  Follow training : tail -f $TRAIN_LOG"
echo "  Follow tuning   : tail -f $TUNE_LOG"
echo "  Check errors    : tail -f $ERR_FILE"
echo ""
echo "  Wait for finish : wait $BG_PID && echo 'DONE'"
echo "========================================================"

# ── Save job metadata ────────────────────────────────────────────────────────
cat > "$LOG_DIR/${JOB_ID}.info" <<INFO
job_id        : $JOB_ID
pid           : $BG_PID
train_csv     : $TRAIN_CSV
test_csv      : $TEST_CSV
n_trials      : $N_TRIALS
text_col      : $TEXT_COL
val_size      : $VAL_SIZE
random_state  : $RANDOM_STATE
mlp_epochs    : $MLP_EPOCHS
mlp_batch_size: $MLP_BATCH_SIZE
mlp_lr        : $MLP_LR
mlp_hidden_dim: $MLP_HIDDEN_DIM
launched      : $(date -u +"%Y-%m-%dT%H:%M:%SZ")
train_log     : $TRAIN_LOG
tune_log      : $TUNE_LOG
err_file      : $ERR_FILE
train_result  : $TRAIN_RESULT
tune_result   : $TUNE_RESULT
INFO
