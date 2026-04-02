#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

SLURM_FILE="${SLURM_FILE:-submit_new.slurm}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results}"
SWEEP_DIR="${SWEEP_DIR:-${RESULTS_DIR}/sweep}"
SBATCH_USER="${SBATCH_USER:-${USER:-kimsh}}"
QUEUE_LIMIT="${QUEUE_LIMIT:-2}"
POLL_SECONDS="${POLL_SECONDS:-30}"
TOP_K="${TOP_K:-10}"
DRY_RUN=0
ANALYZE_ONLY=0
MATRIX_FILE="${MATRIX_FILE:-}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-sweep}"
MANIFEST_FILE="${SWEEP_DIR}/submitted_runs.tsv"

usage() {
  cat <<'EOF'
Usage:
  bash sweep_submit.sh
  bash sweep_submit.sh --dry-run
  bash sweep_submit.sh --analyze-only
  bash sweep_submit.sh --matrix sweep_matrix.txt --user kimsh --queue-limit 2

What it does:
  1. Submits multiple sbatch jobs through submit_new.slurm
  2. Before each submit, checks `squeue -u <user>`
  3. If active jobs >= queue limit, waits and polls
  4. After submission, monitors submitted jobs via squeue
  5. Parses results/slurm_new_*.log into CSV + plots
  6. Writes the best run summary

Matrix format:
  One run per line, comments allowed with '#'.
  Example line:
    RUN_NAME=bert_ft_l2 CLASSIFIER=transformer MODEL=bert START_PHASE=2 INPUT_MODE=text UNFREEZE_LAYERS=2

If no matrix file is provided, a built-in starter sweep is used.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --analyze-only)
      ANALYZE_ONLY=1
      ;;
    --matrix)
      MATRIX_FILE="$2"
      shift
      ;;
    --user)
      SBATCH_USER="$2"
      shift
      ;;
    --queue-limit)
      QUEUE_LIMIT="$2"
      shift
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift
      ;;
    --top-k)
      TOP_K="$2"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

mkdir -p "${SWEEP_DIR}"

declare -a RUN_MATRIX=()

load_default_matrix() {
  RUN_MATRIX=(
    "RUN_NAME=bert_linear CLASSIFIER=linear MODEL=bert START_PHASE=1"
    "RUN_NAME=bert_ft_l2 CLASSIFIER=transformer MODEL=bert START_PHASE=2 INPUT_MODE=text UNFREEZE_LAYERS=2 HEAD_TYPE=mlp LOSS_NAME=cross_entropy"
    "RUN_NAME=bert_ft_l4 CLASSIFIER=transformer MODEL=bert START_PHASE=2 INPUT_MODE=text UNFREEZE_LAYERS=4 HEAD_TYPE=mlp LOSS_NAME=cross_entropy"
    "RUN_NAME=bert_ft_focal CLASSIFIER=transformer MODEL=bert START_PHASE=2 INPUT_MODE=text UNFREEZE_LAYERS=2 HEAD_TYPE=mlp LOSS_NAME=focal"
    "RUN_NAME=roberta_ft_l2 CLASSIFIER=transformer MODEL=roberta START_PHASE=1 INPUT_MODE=text UNFREEZE_LAYERS=2 HEAD_TYPE=mlp LOSS_NAME=cross_entropy"
    "RUN_NAME=bertweet_ft_l2 CLASSIFIER=transformer MODEL=bertweet START_PHASE=1 INPUT_MODE=text UNFREEZE_LAYERS=2 HEAD_TYPE=mlp LOSS_NAME=cross_entropy"
  )
}

load_matrix_from_file() {
  local path="$1"
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [ -z "$line" ] || [[ "$line" == \#* ]]; then
      continue
    fi
    RUN_MATRIX+=("$line")
  done < "$path"
}

extract_token_value() {
  local spec="$1"
  local key="$2"
  local token
  for token in $spec; do
    case "$token" in
      "${key}"=*)
        echo "${token#${key}=}"
        return 0
        ;;
    esac
  done
  return 1
}

active_job_count() {
  local count
  count="$(squeue -u "${SBATCH_USER}" -h 2>/dev/null | wc -l | tr -d ' ')"
  echo "${count:-0}"
}

wait_for_slot() {
  while true; do
    local count
    count="$(active_job_count)"
    echo "[queue] active jobs for ${SBATCH_USER}: ${count} / limit ${QUEUE_LIMIT}"
    if [ "${count}" -lt "${QUEUE_LIMIT}" ]; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

wait_for_visibility() {
  local job_id="$1"
  local attempts=0
  while [ "${attempts}" -lt 20 ]; do
    local state
    state="$(squeue -h -j "${job_id}" -o '%T' 2>/dev/null | head -n 1 || true)"
    if [ -n "${state}" ]; then
      echo "[submit] job ${job_id} visible in squeue with state=${state}"
      return 0
    fi
    attempts=$((attempts + 1))
    sleep 2
  done
  echo "[submit] job ${job_id} not yet visible in squeue; continuing anyway"
}

declare -a SUBMITTED_JOB_IDS=()

submit_run() {
  local spec="$1"
  local index="$2"
  local run_name
  run_name="$(extract_token_value "${spec}" RUN_NAME || true)"
  if [ -z "${run_name}" ]; then
    run_name="run_$(printf '%02d' "${index}")"
    spec="RUN_NAME=${run_name} ${spec}"
  fi

  wait_for_slot

  local job_name="${JOB_NAME_PREFIX}-${run_name}"
  local cmd=(sbatch --job-name "${job_name}" --export "ALL,${spec}" "${SLURM_FILE}")

  echo
  echo "[submit] ${run_name}"
  echo "[submit] spec: ${spec}"
  echo "[submit] command: ${cmd[*]}"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  local out
  out="$("${cmd[@]}")"
  echo "[submit] ${out}"

  local job_id
  job_id="$(echo "${out}" | awk '/Submitted batch job/ {print $4}')"
  if [ -z "${job_id}" ]; then
    echo "Failed to parse sbatch output: ${out}" >&2
    exit 1
  fi

  SUBMITTED_JOB_IDS+=("${job_id}")
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${run_name}" "${job_name}" "${spec}" >> "${MANIFEST_FILE}"
  wait_for_visibility "${job_id}"
}

monitor_jobs() {
  if [ "${DRY_RUN}" = "1" ] || [ "${#SUBMITTED_JOB_IDS[@]}" -eq 0 ]; then
    return 0
  fi

  echo
  echo "[monitor] polling submitted jobs until they leave squeue ..."
  while true; do
    local active=0
    local job_id
    echo "[monitor] $(date '+%Y-%m-%d %H:%M:%S')"
    for job_id in "${SUBMITTED_JOB_IDS[@]}"; do
      local row
      row="$(squeue -h -j "${job_id}" -o '%i|%T|%M|%R' 2>/dev/null || true)"
      if [ -n "${row}" ]; then
        active=1
        echo "  ${row}"
      fi
    done
    if [ "${active}" -eq 0 ]; then
      echo "[monitor] all submitted jobs have left squeue."
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

analyze_logs() {
  python - "${RESULTS_DIR}" "${SWEEP_DIR}" "${TOP_K}" <<'PY'
import csv
import glob
import math
import os
import re
import shlex
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"[analyze] plotting libraries unavailable: {exc}")
    pd = None
    plt = None


RESULTS_DIR = sys.argv[1]
SWEEP_DIR = sys.argv[2]
TOP_K = int(sys.argv[3])

os.makedirs(SWEEP_DIR, exist_ok=True)


def to_float(value):
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_command_args(command):
    args = {}
    if not command:
        return args
    tokens = shlex.split(command)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                args[key] = tokens[i + 1]
                i += 2
            else:
                args[key] = "1"
                i += 1
        else:
            i += 1
    return args


def search(pattern, text, flags=0):
    match = re.search(pattern, text, flags)
    return match.groups() if match else None


def build_label(row):
    if row["classifier"] == "linear":
        return f"{row['model']} | linear"
    pieces = [
        row["model"],
        row.get("input_mode") or "text",
        f"L{row.get('unfreeze_layers') or '?'}",
        row.get("head_type") or "head",
        row.get("loss_name") or "loss",
    ]
    return " | ".join(pieces)


def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if "Sentiment Classification Pipeline" not in text:
        return None

    job_id = search(r"^Job ID:\s+(\d+)", text, re.M)
    command = search(r"^Command:\s+(.*)$", text, re.M)
    run_name = search(r"^Run name:\s+(.*)$", text, re.M)
    classifier = search(r"^Classifier:\s+(\S+)", text, re.M)
    model_top = search(r"^Model:\s+(.*?)\s+\(--model\s+([^)]+)\)", text, re.M)
    args = parse_command_args(command[0] if command else "")

    row = {
        "log_path": path,
        "job_id": job_id[0] if job_id else "",
        "run_name": run_name[0].strip() if run_name else "",
        "command": command[0] if command else "",
        "classifier": args.get("classifier") or (classifier[0] if classifier else ""),
        "model": args.get("model") or (model_top[1].strip() if model_top else ""),
        "model_name": model_top[0].strip() if model_top else "",
        "start_phase": args.get("start_phase", ""),
        "input_mode": args.get("input_mode", ""),
        "head_type": args.get("head_type", ""),
        "loss_name": args.get("loss_name", ""),
        "unfreeze_layers": args.get("unfreeze_layers", ""),
        "epochs": args.get("epochs", ""),
        "pooling": args.get("pooling", ""),
        "encoder_lr": args.get("encoder_lr", ""),
        "classifier_lr": args.get("classifier_lr", ""),
        "use_time_of_tweet": args.get("use_time_of_tweet", ""),
        "use_age_of_user": args.get("use_age_of_user", ""),
        "use_country": args.get("use_country", ""),
    }

    transformer_metrics = search(
        r"Best checkpoint test:\s+loss=([0-9.]+)\s+macro_f1=([0-9.]+)\s+acc=([0-9.]+)",
        text,
    )
    transformer_val = search(r"Best epoch:\s+(\d+)\s+\|\s+best val_f1=([0-9.]+)", text)
    linear_metrics = search(r"Test macro-F1:\s*([0-9.]+)\s*\|\s*accuracy=([0-9.]+)", text)
    linear_val = search(r"Best val macro-F1:\s*([0-9.]+)", text)

    if transformer_metrics:
        row["status"] = "complete"
        row["best_epoch"] = transformer_val[0] if transformer_val else ""
        row["val_macro_f1"] = transformer_val[1] if transformer_val else ""
        row["test_loss"] = transformer_metrics[0]
        row["test_macro_f1"] = transformer_metrics[1]
        row["test_accuracy"] = transformer_metrics[2]
    elif linear_metrics:
        row["status"] = "complete"
        row["best_epoch"] = ""
        row["val_macro_f1"] = linear_val[0] if linear_val else ""
        row["test_loss"] = ""
        row["test_macro_f1"] = linear_metrics[0]
        row["test_accuracy"] = linear_metrics[1]
    else:
        row["status"] = "incomplete"
        row["best_epoch"] = ""
        row["val_macro_f1"] = ""
        row["test_loss"] = ""
        row["test_macro_f1"] = ""
        row["test_accuracy"] = ""

    row["label"] = build_label(row)
    return row


rows = []
for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "slurm_new_*.log"))):
    parsed = parse_log(path)
    if parsed is not None:
        rows.append(parsed)

summary_csv = os.path.join(SWEEP_DIR, "sweep_summary.csv")
fieldnames = [
    "job_id",
    "run_name",
    "status",
    "classifier",
    "model",
    "model_name",
    "input_mode",
    "unfreeze_layers",
    "head_type",
    "loss_name",
    "epochs",
    "pooling",
    "encoder_lr",
    "classifier_lr",
    "val_macro_f1",
    "test_loss",
    "test_macro_f1",
    "test_accuracy",
    "best_epoch",
    "start_phase",
    "use_time_of_tweet",
    "use_age_of_user",
    "use_country",
    "label",
    "log_path",
    "command",
]

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

complete_rows = [row for row in rows if row.get("status") == "complete" and to_float(row.get("test_macro_f1")) is not None]
complete_rows.sort(
    key=lambda row: (
        to_float(row.get("test_macro_f1")) or float("-inf"),
        to_float(row.get("val_macro_f1")) or float("-inf"),
    ),
    reverse=True,
)

best_txt = os.path.join(SWEEP_DIR, "best_run.txt")
top_txt = os.path.join(SWEEP_DIR, "top_runs.txt")

with open(top_txt, "w", encoding="utf-8") as f:
    if not complete_rows:
        f.write("No complete runs were parsed.\n")
    else:
        for idx, row in enumerate(complete_rows[:TOP_K], 1):
            f.write(
                f"{idx}. {row['label']} | test_f1={row['test_macro_f1']} | "
                f"val_f1={row['val_macro_f1']} | acc={row['test_accuracy']} | "
                f"log={row['log_path']}\n"
            )

with open(best_txt, "w", encoding="utf-8") as f:
    if not complete_rows:
        f.write("No complete runs were parsed.\n")
    else:
        best = complete_rows[0]
        f.write("Best run\n")
        f.write(f"job_id: {best['job_id']}\n")
        f.write(f"run_name: {best['run_name']}\n")
        f.write(f"label: {best['label']}\n")
        f.write(f"classifier: {best['classifier']}\n")
        f.write(f"model: {best['model']}\n")
        f.write(f"test_macro_f1: {best['test_macro_f1']}\n")
        f.write(f"val_macro_f1: {best['val_macro_f1']}\n")
        f.write(f"test_accuracy: {best['test_accuracy']}\n")
        f.write(f"log_path: {best['log_path']}\n")
        f.write(f"command: {best['command']}\n")

print(f"[analyze] wrote {summary_csv}")
print(f"[analyze] wrote {top_txt}")
print(f"[analyze] wrote {best_txt}")
if complete_rows:
    best = complete_rows[0]
    print(
        "[analyze] best run: "
        f"{best['label']} | test_f1={best['test_macro_f1']} | "
        f"val_f1={best['val_macro_f1']} | acc={best['test_accuracy']}"
    )
else:
    print("[analyze] no complete runs found")

if pd is None or plt is None or not complete_rows:
    sys.exit(0)

df = pd.DataFrame(complete_rows)
for col in ["test_macro_f1", "val_macro_f1", "test_accuracy", "test_loss"]:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values("test_macro_f1", ascending=True)

plt.figure(figsize=(12, max(6, 0.35 * len(df))))
plt.barh(df["label"], df["test_macro_f1"], color="#4C78A8")
plt.xlabel("Test Macro-F1")
plt.title("Sweep Runs Sorted By Test Macro-F1")
plt.tight_layout()
all_runs_path = os.path.join(SWEEP_DIR, "all_runs_test_macro_f1.png")
plt.savefig(all_runs_path, dpi=200)
plt.close()
print(f"[analyze] wrote {all_runs_path}")

best_by_model = (
    df.groupby("model", as_index=False)["test_macro_f1"]
      .max()
      .sort_values("test_macro_f1", ascending=False)
)
if not best_by_model.empty:
    plt.figure(figsize=(10, max(4, 0.5 * len(best_by_model))))
    plt.bar(best_by_model["model"], best_by_model["test_macro_f1"], color="#72B7B2")
    plt.ylabel("Best Test Macro-F1")
    plt.title("Best Test Macro-F1 By Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    model_plot = os.path.join(SWEEP_DIR, "best_by_model.png")
    plt.savefig(model_plot, dpi=200)
    plt.close()
    print(f"[analyze] wrote {model_plot}")

for field in ["classifier", "model", "input_mode", "unfreeze_layers", "head_type", "loss_name", "pooling"]:
    if field not in df.columns:
        continue
    sub = df[[field, "test_macro_f1"]].dropna()
    if sub.empty:
        continue
    unique_vals = [str(v) for v in sub[field].unique() if str(v).strip()]
    if not (1 < len(unique_vals) <= 12):
        continue
    agg = sub.groupby(field, as_index=False)["test_macro_f1"].max().sort_values("test_macro_f1", ascending=False)
    plt.figure(figsize=(10, max(4, 0.5 * len(agg))))
    plt.bar(agg[field].astype(str), agg["test_macro_f1"], color="#F58518")
    plt.ylabel("Best Test Macro-F1")
    plt.title(f"Best Test Macro-F1 By {field.replace('_', ' ').title()}")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out_path = os.path.join(SWEEP_DIR, f"best_by_{field}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[analyze] wrote {out_path}")
PY
}

if [ "${ANALYZE_ONLY}" = "0" ]; then
  if [ ! -f "${SLURM_FILE}" ]; then
    echo "SLURM file not found: ${SLURM_FILE}" >&2
    exit 1
  fi

  if [ -n "${MATRIX_FILE}" ]; then
    if [ ! -f "${MATRIX_FILE}" ]; then
      echo "Matrix file not found: ${MATRIX_FILE}" >&2
      exit 1
    fi
    load_matrix_from_file "${MATRIX_FILE}"
  else
    load_default_matrix
  fi

  if [ "${#RUN_MATRIX[@]}" -eq 0 ]; then
    echo "No sweep runs configured." >&2
    exit 1
  fi

  printf 'job_id\trun_name\tjob_name\tspec\n' > "${MANIFEST_FILE}"

  echo "[sweep] using SLURM file: ${SLURM_FILE}"
  echo "[sweep] results dir: ${RESULTS_DIR}"
  echo "[sweep] sweep dir: ${SWEEP_DIR}"
  echo "[sweep] user: ${SBATCH_USER}"
  echo "[sweep] queue limit: ${QUEUE_LIMIT}"
  echo "[sweep] poll seconds: ${POLL_SECONDS}"
  echo "[sweep] configured runs: ${#RUN_MATRIX[@]}"

  index=1
  for spec in "${RUN_MATRIX[@]}"; do
    submit_run "${spec}" "${index}"
    index=$((index + 1))
  done

  monitor_jobs
fi

analyze_logs
