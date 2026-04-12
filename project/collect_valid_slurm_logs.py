#!/usr/bin/env python3
"""Copy valid Slurm logs into a portable candidate directory.

Default behavior is intentionally conservative:
  - scan results/slurm_new_*.log
  - keep only completed full-pipeline runs
  - copy the .log and matching .err into ../candidate
  - write manifest.tsv and manifest.jsonl for later cross-comparison

Run from the project root on the cluster:
  python collect_valid_slurm_logs.py

Useful variants:
  python collect_valid_slurm_logs.py --dry-run
  python collect_valid_slurm_logs.py --allow-resume
  python collect_valid_slurm_logs.py --out-dir ../candidate_roberta --model roberta-large
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LogRecord:
    job_id: str
    status: str
    valid: bool
    reason: str
    model_key: str
    hf_model: str
    classifier: str
    include_d25: str
    include_d4: str
    phases: str
    elapsed_min: str
    node: str
    time: str
    gpu: str
    result_path: str
    log_path: str
    err_path: str
    copied_log: str
    copied_err: str


def _first(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def parse_log(log_path: Path) -> LogRecord:
    text = log_path.read_text(errors="ignore")
    job_id = _first(r"Job ID:\s*(\d+)", text, log_path.stem.split("_")[-1])
    node = _first(r"Node:\s*(.+)", text)
    time = _first(r"Time:\s*(.+)", text)
    gpu = _first(r"GPU:\s*(.+)", text)
    elapsed_min = _first(r"All phases complete in\s*([\d.]+)\s*minutes", text)
    result_path = _first(r"Results:\s*(.+)", text)
    phases = _first(r"Running phases:\s*(.+)", text)
    classifier = _first(r"Classifier:\s*(\S+)", text)
    include_d25 = _first(r"Include D2\.5:\s*(\S+)", text)
    include_d4 = _first(r"Include D4:\s*(\S+)", text)

    model_match = re.search(r"Model:\s*(.*?)\s+\(--model\s+([^)]+)\)", text)
    hf_model = model_match.group(1).strip() if model_match else ""
    model_key = model_match.group(2).strip() if model_match else ""

    # Fallback for logs that failed before the run_all banner.
    if not model_key:
        run_match = re.search(r"Running phases from .* classifier=(\S+) model=(\S+)", text)
        if run_match:
            classifier = classifier or run_match.group(1).strip()
            model_key = run_match.group(2).strip()
    if not model_key:
        only_match = re.search(r"Running only phase .* classifier=(\S+) model=(\S+)", text)
        if only_match:
            classifier = classifier or only_match.group(1).strip()
            model_key = only_match.group(2).strip()

    stopped_phase = _first(r"Pipeline stopped at phase\s*(\d+)", text)
    if elapsed_min:
        status = "completed"
    elif stopped_phase:
        status = f"failed_phase_{stopped_phase}"
    elif "Traceback" in text or "[ERROR]" in text:
        status = "failed"
    else:
        status = "unknown"

    err_path = log_path.with_suffix(".err")
    if not err_path.exists():
        err_path = log_path.with_name(log_path.name.replace(".log", ".err"))

    return LogRecord(
        job_id=job_id,
        status=status,
        valid=False,
        reason="unclassified",
        model_key=model_key,
        hf_model=hf_model,
        classifier=classifier,
        include_d25=include_d25,
        include_d4=include_d4,
        phases=phases,
        elapsed_min=elapsed_min,
        node=node,
        time=time,
        gpu=gpu,
        result_path=result_path,
        log_path=str(log_path),
        err_path=str(err_path) if err_path.exists() else "",
        copied_log="",
        copied_err="",
    )


def mark_valid(
    rec: LogRecord,
    *,
    allow_resume: bool,
    allow_only_phase: bool,
    model_filter: str | None,
    classifier_filter: str | None,
) -> LogRecord:
    if rec.status != "completed":
        rec.reason = rec.status
        return rec

    if model_filter and rec.model_key != model_filter:
        rec.reason = f"model_mismatch:{rec.model_key}"
        return rec

    if classifier_filter and rec.classifier != classifier_filter:
        rec.reason = f"classifier_mismatch:{rec.classifier}"
        return rec

    normalized_phases = re.sub(r"\s+", "", rec.phases)
    is_full_pipeline = normalized_phases == "[1,2,3,4]"
    is_resume_like = normalized_phases in {"[2,3,4]", "[3,4]", "[4]"}
    is_only_phase = bool(normalized_phases and not is_full_pipeline and not is_resume_like)

    if is_full_pipeline:
        rec.valid = True
        rec.reason = "completed_full_pipeline"
    elif allow_resume and is_resume_like:
        rec.valid = True
        rec.reason = f"completed_resume:{rec.phases}"
    elif allow_only_phase and is_only_phase:
        rec.valid = True
        rec.reason = f"completed_only_or_custom:{rec.phases}"
    else:
        rec.reason = f"completed_but_not_full_pipeline:{rec.phases or 'unknown_phases'}"

    return rec


def write_manifest(records: list[LogRecord], out_dir: Path) -> None:
    fields = list(asdict(records[0]).keys()) if records else list(LogRecord.__dataclass_fields__.keys())
    tsv_path = out_dir / "manifest.tsv"
    jsonl_path = out_dir / "manifest.jsonl"

    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for rec in records:
            row = [str(asdict(rec)[field]).replace("\t", " ") for field in fields]
            f.write("\t".join(row) + "\n")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy completed Slurm logs to ../candidate.")
    parser.add_argument("--results-dir", default="results", help="Directory containing slurm_new_*.log files.")
    parser.add_argument("--out-dir", default="../candidate", help="Destination directory for copied logs.")
    parser.add_argument("--pattern", default="slurm_new_*.log", help="Log glob pattern inside --results-dir.")
    parser.add_argument("--allow-resume", action="store_true", help="Also accept completed resumed runs, e.g. phases [2, 3, 4].")
    parser.add_argument("--allow-only-phase", action="store_true", help="Also accept completed only/custom phase runs.")
    parser.add_argument("--model", default=None, help="Optional exact --model key filter, e.g. roberta-large.")
    parser.add_argument("--classifier", default=None, help="Optional classifier filter, e.g. prototype.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print without copying.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"

    log_paths = sorted(results_dir.glob(args.pattern))
    if not log_paths:
        print(f"No logs matched: {results_dir / args.pattern}")
        return

    records: list[LogRecord] = []
    for log_path in log_paths:
        rec = parse_log(log_path)
        rec = mark_valid(
            rec,
            allow_resume=args.allow_resume,
            allow_only_phase=args.allow_only_phase,
            model_filter=args.model,
            classifier_filter=args.classifier,
        )
        records.append(rec)

    valid_records = [rec for rec in records if rec.valid]

    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
        for rec in valid_records:
            src_log = Path(rec.log_path)
            dst_log = logs_dir / src_log.name
            shutil.copy2(src_log, dst_log)
            rec.copied_log = str(dst_log)

            if rec.err_path:
                src_err = Path(rec.err_path)
                if src_err.exists():
                    dst_err = logs_dir / src_err.name
                    shutil.copy2(src_err, dst_err)
                    rec.copied_err = str(dst_err)

        write_manifest(records, out_dir)

    print(f"Scanned logs: {len(records)}")
    print(f"Valid logs:   {len(valid_records)}")
    print(f"Rejected:     {len(records) - len(valid_records)}")
    print(f"Output dir:   {out_dir}")
    if args.dry_run:
        print("Dry run only; nothing copied.")

    print("\nValid jobs:")
    for rec in valid_records:
        print(
            f"  {rec.job_id}\t{rec.model_key or '?'}\t{rec.classifier or '?'}"
            f"\tD25={rec.include_d25 or '?'}\tD4={rec.include_d4 or '?'}"
            f"\t{rec.elapsed_min or '?'} min\t{rec.reason}"
        )

    rejected_preview = [rec for rec in records if not rec.valid][:20]
    if rejected_preview:
        print("\nRejected preview:")
        for rec in rejected_preview:
            print(f"  {rec.job_id}\t{rec.model_key or '?'}\t{rec.status}\t{rec.reason}")


if __name__ == "__main__":
    main()
