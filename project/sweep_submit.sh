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
SWEEP_PRESET="${SWEEP_PRESET:-full}"
INCLUDE_SELECTED_INPUTS="${INCLUDE_SELECTED_INPUTS:-0}"
INCLUDE_LLM_MODELS="${INCLUDE_LLM_MODELS:-0}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-sweep}"
MANIFEST_FILE="${SWEEP_DIR}/submitted_runs.tsv"

usage() {
  cat <<'EOF'
Usage:
  bash sweep_submit.sh
  bash sweep_submit.sh --dry-run
  bash sweep_submit.sh --analyze-only
  bash sweep_submit.sh --preset full
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

Built-in presets:
  starter    Small sanity sweep.
  full       Broad practical sweep across models, unfreeze depth, head, loss, and metadata.
  exhaustive Larger combinatorial sweep; intended for long overnight runs.

Notes:
  - If no matrix file is provided, the built-in preset is used. Default preset: full.
  - Built-in presets use START_PHASE=1 for correctness and isolate each run by RUN_NAME.
  - selected_text-based input modes are excluded by default because fixed test.csv lacks selected_text.
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
    --preset)
      SWEEP_PRESET="$2"
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
    --include-selected-inputs)
      INCLUDE_SELECTED_INPUTS=1
      ;;
    --include-llms)
      INCLUDE_LLM_MODELS=1
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

PYTHON_BIN_RESOLVED="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN_RESOLVED}" ]; then
  PYTHON_BIN_RESOLVED="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
fi
if [ -z "${PYTHON_BIN_RESOLVED}" ]; then
  echo "[ERROR] No Python interpreter found for sweep_submit.sh." >&2
  echo "[ERROR] Activate your env first, or export PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

generate_preset_matrix() {
  local preset="$1"
  local generated
  generated="$(
    SWEEP_PRESET_VALUE="${preset}" \
    INCLUDE_SELECTED_INPUTS="${INCLUDE_SELECTED_INPUTS}" \
    INCLUDE_LLM_MODELS="${INCLUDE_LLM_MODELS}" \
    "${PYTHON_BIN_RESOLVED}" - <<'PY'
import itertools
import os

preset = os.environ.get("SWEEP_PRESET_VALUE", "full")
include_selected = os.environ.get("INCLUDE_SELECTED_INPUTS", "0") == "1"
include_llms = os.environ.get("INCLUDE_LLM_MODELS", "0") == "1"

practical_models = ["bert", "distilbert", "roberta", "bertweet", "finbert"]
llm_models = ["qwen2", "tinyllama"] if include_llms else []
models = practical_models + llm_models

meta_presets = {
    "none": {},
    "time": {"USE_TIME_OF_TWEET": "1"},
    "age": {"USE_AGE_OF_USER": "1"},
    "country": {"USE_COUNTRY": "1"},
    "time_age": {"USE_TIME_OF_TWEET": "1", "USE_AGE_OF_USER": "1"},
    "time_country": {"USE_TIME_OF_TWEET": "1", "USE_COUNTRY": "1"},
    "age_country": {"USE_AGE_OF_USER": "1", "USE_COUNTRY": "1"},
    "all": {"USE_TIME_OF_TWEET": "1", "USE_AGE_OF_USER": "1", "USE_COUNTRY": "1"},
}

lr_presets = {
    "std": {"ENCODER_LR": "2e-5", "CLASSIFIER_LR": "1e-4"},
    "low": {"ENCODER_LR": "1e-5", "CLASSIFIER_LR": "5e-5"},
}

def emit(spec):
    print(" ".join(f"{k}={v}" for k, v in spec.items()))

def build_run_name(model, input_mode, unfreeze, head, loss, meta_name, lr_name=None, pooling=None):
    pieces = [model]
    if input_mode != "text":
        pieces.append(input_mode.replace("text_", "").replace("selected_", "sel_"))
    pieces.append(f"u{unfreeze}")
    pieces.append("lin" if head == "linear" else "mlp")
    pieces.append("ce" if loss == "cross_entropy" else "focal")
    if meta_name != "none":
        pieces.append(f"meta_{meta_name}")
    if lr_name:
        pieces.append(lr_name)
    if pooling and pooling != "auto":
        pieces.append(pooling)
    return "_".join(pieces)

def add_linear_runs():
    for model in models:
        emit({
            "RUN_NAME": f"{model}_linear",
            "CLASSIFIER": "linear",
            "MODEL": model,
            "START_PHASE": "1",
        })

def add_transformer_runs(unfreeze_values, head_types, loss_names, metadata_names, input_modes=None, lr_names=None, poolings=None):
    input_modes = input_modes or ["text"]
    lr_names = lr_names or ["std"]
    poolings = poolings or ["auto"]
    for model, input_mode, unfreeze, head, loss, meta_name, lr_name, pooling in itertools.product(
        models, input_modes, unfreeze_values, head_types, loss_names, metadata_names, lr_names, poolings
    ):
        spec = {
            "RUN_NAME": build_run_name(model, input_mode, unfreeze, head, loss, meta_name, lr_name, pooling),
            "CLASSIFIER": "transformer",
            "MODEL": model,
            "START_PHASE": "1",
            "INPUT_MODE": input_mode,
            "UNFREEZE_LAYERS": str(unfreeze),
            "HEAD_TYPE": head,
            "LOSS_NAME": loss,
            "POOLING": pooling,
        }
        spec.update(lr_presets[lr_name])
        spec.update(meta_presets[meta_name])
        emit(spec)

if preset == "starter":
    add_linear_runs()
    add_transformer_runs(
        unfreeze_values=[2, 4],
        head_types=["mlp"],
        loss_names=["cross_entropy"],
        metadata_names=["none"],
        input_modes=["text"],
    )
elif preset == "full":
    add_linear_runs()
    add_transformer_runs(
        unfreeze_values=[0, 2, 4, 12],
        head_types=["linear", "mlp"],
        loss_names=["cross_entropy", "focal"],
        metadata_names=["none", "all"],
        input_modes=["text"],
    )
elif preset == "exhaustive":
    add_linear_runs()
    input_modes = ["text"]
    if include_selected:
        input_modes.extend(["text_plus_selected", "text_selected_pair"])
    add_transformer_runs(
        unfreeze_values=[0, 2, 4, 12],
        head_types=["linear", "mlp"],
        loss_names=["cross_entropy", "focal"],
        metadata_names=list(meta_presets.keys()),
        input_modes=input_modes,
        lr_names=["std", "low"],
        poolings=["auto", "mean"],
    )
else:
    raise SystemExit(f"Unknown sweep preset: {preset}")
PY
  )"
  RUN_MATRIX=()
  while IFS= read -r line || [ -n "$line" ]; do
    if [ -n "$line" ]; then
      RUN_MATRIX+=("$line")
    fi
  done <<< "${generated}"
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
  if ! command -v squeue >/dev/null 2>&1; then
    echo "[ERROR] squeue is not available in PATH." >&2
    exit 1
  fi
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

  local job_name="${JOB_NAME_PREFIX}-${run_name}"
  local slurm_path="${SLURM_FILE}"
  if [[ "${slurm_path}" != /* ]]; then
    slurm_path="${PROJECT_DIR}/${SLURM_FILE}"
  fi
  if [ ! -f "${slurm_path}" ]; then
    echo "[submit][ERROR] SLURM file not found: ${slurm_path}" >&2
    exit 1
  fi

  local cmd=(sbatch --chdir "${PROJECT_DIR}" --job-name "${job_name}" --export "ALL,${spec}" "${slurm_path}")

  echo
  echo "[submit] ${run_name}"
  echo "[submit] spec: ${spec}"
  echo "[submit] command: ${cmd[*]}"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  wait_for_slot

  if ! command -v sbatch >/dev/null 2>&1; then
    echo "[submit][ERROR] sbatch is not available in PATH." >&2
    exit 1
  fi

  local out
  if ! out="$("${cmd[@]}" 2>&1)"; then
    echo "[submit][ERROR] sbatch failed for run '${run_name}'" >&2
    echo "[submit][ERROR] spec: ${spec}" >&2
    echo "${out}" >&2
    exit 1
  fi
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
  "${PYTHON_BIN_RESOLVED}" - "${RESULTS_DIR}" "${SWEEP_DIR}" "${TOP_K}" "${MANIFEST_FILE}" <<'PY'
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
MANIFEST_FILE = sys.argv[4]

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


def load_manifest_job_ids(path):
    if not path or not os.path.exists(path):
        return []
    job_ids = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if idx == 0 and line.startswith("job_id\t"):
                continue
            parts = line.split("\t")
            if parts and parts[0].strip():
                job_ids.append(parts[0].strip())
    return job_ids


def detect_split_regime(text):
    if (
        "using fixed test split from 'test.csv'" in text
        or "using fixed Kaggle test split from 'test.csv'" in text
        or "Loading local test dataset from" in text
        or "Loading local test dataset" in text
    ):
        return "fixed_test_csv"
    if "Loading dataset '" in text and re.search(r"train=\d+\s+\|\s+val=\d+\s+\|\s+test=\d+", text):
        return "dataset_native_split"
    if re.search(r"split -> train=\d+\s+\|\s+val=\d+\s+\|\s+test=\d+", text):
        return "random_split"
    return "unknown"


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
    data_source = search(r"^Data source:\s+(.*)$", text, re.M)
    split_sizes = search(r"^Split sizes:\s+train=(\d+)\s+\|\s+val=(\d+)\s+\|\s+test=(\d+)", text, re.M)
    raw_split = search(r"^\s*split -> train=(\d+)\s+\|\s+val=(\d+)(?:\s+\|\s+test=(\d+))?", text, re.M)
    split_regime = detect_split_regime(text)

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
        "data_source": data_source[0].strip() if data_source else "",
        "split_regime": split_regime,
        "uses_fixed_test_csv": "1" if split_regime == "fixed_test_csv" else "0",
        "train_size": "",
        "val_size": "",
        "test_size": "",
    }

    if split_sizes:
        row["train_size"], row["val_size"], row["test_size"] = split_sizes
    elif raw_split:
        row["train_size"] = raw_split[0]
        row["val_size"] = raw_split[1]
        row["test_size"] = raw_split[2] if len(raw_split) > 2 and raw_split[2] else ""

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


manifest_job_ids = load_manifest_job_ids(MANIFEST_FILE)
manifest_paths = [
    os.path.join(RESULTS_DIR, f"slurm_new_{job_id}.log")
    for job_id in manifest_job_ids
]
candidate_paths = [path for path in manifest_paths if os.path.exists(path)]
analysis_scope = "manifest" if candidate_paths else "all_logs"
if not candidate_paths:
    candidate_paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "slurm_new_*.log")))

rows = []
for path in candidate_paths:
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
    "data_source",
    "split_regime",
    "uses_fixed_test_csv",
    "train_size",
    "val_size",
    "test_size",
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
ranked_rows = complete_rows
ranking_note = "ranking all parsed complete runs"
fixed_test_rows = [row for row in complete_rows if row.get("split_regime") == "fixed_test_csv"]
if fixed_test_rows:
    ranked_rows = fixed_test_rows
    ranking_note = "ranking only fixed_test_csv runs"

ranked_rows.sort(
    key=lambda row: (
        to_float(row.get("test_macro_f1")) or float("-inf"),
        to_float(row.get("val_macro_f1")) or float("-inf"),
    ),
    reverse=True,
)

best_txt = os.path.join(SWEEP_DIR, "best_run.txt")
top_txt = os.path.join(SWEEP_DIR, "top_runs.txt")

with open(top_txt, "w", encoding="utf-8") as f:
    f.write(f"analysis_scope: {analysis_scope}\n")
    f.write(f"{ranking_note}\n\n")
    if not ranked_rows:
        f.write("No complete runs were parsed.\n")
    else:
        for idx, row in enumerate(ranked_rows[:TOP_K], 1):
            f.write(
                f"{idx}. {row['label']} | test_f1={row['test_macro_f1']} | "
                f"val_f1={row['val_macro_f1']} | acc={row['test_accuracy']} | "
                f"split={row['split_regime']} | test_size={row['test_size']} | "
                f"log={row['log_path']}\n"
            )

with open(best_txt, "w", encoding="utf-8") as f:
    f.write(f"analysis_scope: {analysis_scope}\n")
    f.write(f"ranking_note: {ranking_note}\n")
    if not ranked_rows:
        f.write("No complete runs were parsed.\n")
    else:
        best = ranked_rows[0]
        f.write("Best run\n")
        f.write(f"job_id: {best['job_id']}\n")
        f.write(f"run_name: {best['run_name']}\n")
        f.write(f"label: {best['label']}\n")
        f.write(f"classifier: {best['classifier']}\n")
        f.write(f"model: {best['model']}\n")
        f.write(f"data_source: {best['data_source']}\n")
        f.write(f"split_regime: {best['split_regime']}\n")
        f.write(f"uses_fixed_test_csv: {best['uses_fixed_test_csv']}\n")
        f.write(f"test_size: {best['test_size']}\n")
        f.write(f"test_macro_f1: {best['test_macro_f1']}\n")
        f.write(f"val_macro_f1: {best['val_macro_f1']}\n")
        f.write(f"test_accuracy: {best['test_accuracy']}\n")
        f.write(f"log_path: {best['log_path']}\n")
        f.write(f"command: {best['command']}\n")

print(f"[analyze] wrote {summary_csv}")
print(f"[analyze] wrote {top_txt}")
print(f"[analyze] wrote {best_txt}")
print(f"[analyze] analysis scope: {analysis_scope}")
print(f"[analyze] {ranking_note}")
if ranked_rows:
    best = ranked_rows[0]
    print(
        "[analyze] best run: "
        f"{best['label']} | test_f1={best['test_macro_f1']} | "
        f"val_f1={best['val_macro_f1']} | acc={best['test_accuracy']} | "
        f"split={best['split_regime']} | test_size={best['test_size']}"
    )
else:
    print("[analyze] no complete runs found")

if pd is None or plt is None or not ranked_rows:
    sys.exit(0)

df = pd.DataFrame(ranked_rows)
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
    generate_preset_matrix "${SWEEP_PRESET}"
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
  if [ -n "${MATRIX_FILE}" ]; then
    echo "[sweep] matrix source: ${MATRIX_FILE}"
  else
    echo "[sweep] preset: ${SWEEP_PRESET}"
    echo "[sweep] include selected inputs: ${INCLUDE_SELECTED_INPUTS}"
    echo "[sweep] include llm models: ${INCLUDE_LLM_MODELS}"
  fi
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
