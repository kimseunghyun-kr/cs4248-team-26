# Slurm Job ID Sleuth

This note tracks what can and cannot be recovered about experiment job IDs.

## Short Answer

The experiment result files do **not** currently record the Slurm job ID.

The job ID is recorded only in:
- Slurm log filename: `results/slurm_new_<jobid>.log`
- Slurm error filename: `results/slurm_new_<jobid>.err`
- first line of the Slurm log: `Job ID: <jobid>`

So if `results/slurm_new_*.log` / `results/slurm_new_*.err` are deleted or not copied from the cluster, the job ID cannot be reconstructed from `eval_report*.txt` alone.

## Local Repo Status

In the local checkout inspected here:
- `results/slurm_new_*.log` is not present
- `results/slurm_new_*.err` is not present
- `sweep_submit.sh` / `sweep_submit.slurm` were not present at repo root
- `results/eval_report*.txt` contains metrics but not job IDs

Therefore the complete job-ID mapping must be recovered from the cluster-side `results/` directory if those logs still exist.

## Job IDs Seen In Pasted Logs

These are the IDs that appeared directly in copied Slurm snippets during debugging.

| Job ID | Model / run | Status | Notes |
|---:|---|---|---|
| `568225` | likely `gemma4-26b-it` | failed | Early Gemma run; Slurm reported an `oom_kill` event during weight loading / setup. |
| `568365` | `gemma4-26b-it` | failed | Phase 1 load failed. Log showed `google/gemma-4-26B-A4B-it`, loader mismatch symptoms, and CUDA OOM. |
| `568651` | `gemma4-26b-it` | failed | Phase 1 failed with `AttributeError: 'Gemma4Config' object has no attribute 'hidden_size'`. This led to the nested Gemma config/core-model fix. |
| `569355` | `gemma4-26b-it` | failed | Phase 1 succeeded, then Phase 2 failed while loading a second Gemma encoder. This led to explicit per-condition GPU cleanup in `cbdc/refine.py`. |
| `570011` | `roberta-large`, `CLASSIFIER=prototype`, `INCLUDE_D25=1`, `INCLUDE_D4=1` | completed | Node `xgpi6`; full run completed in `24.4` min. Important result: `D2/D2.5` got high accuracy through pathological neutral collapse; `D4` was near `B1`. |
| `570012` | `bert-large-cased`, `CLASSIFIER=prototype`, `INCLUDE_D25=1`, `INCLUDE_D4=1` | completed | Node `xgpi3`; full run completed in `19.7` min. Important result: `D1` was strongest; `D2` near no-op; `D2.5/D3` modest; `D4` near `B1`. |

## Runs Mentioned Without Job ID In Pasted Excerpts

These runs had logs/results pasted, but the pasted excerpt did not include the top `Job ID:` line.

| Run | Status | Notes |
|---|---|---|
| `qwen25-3b`, `CLASSIFIER=prototype`, `INCLUDE_D25=1`, `INCLUDE_D4=1` | completed | Full run completed in `131.9` min. `B1` already had severe neutral collapse; `D4` failed to repair it and slightly worsened macro-F1. |
| `gemma4-26b-it`, `CLASSIFIER=prototype`, likely `INCLUDE_D25=1`, `INCLUDE_D4=1` | completed | Full run completed in `81.4` min. `D1` gave a small gain; `D2/D2.5` roughly flat; `D3/D4` harmful. |

To recover the job IDs for these, check the cluster logs around the corresponding completion times:
- Qwen run finished around `Fri Apr 10 20:04:53 +08 2026`
- Gemma successful run finished around `Fri Apr 10 22:22:33 +08 2026`

## Cluster-Side Parser

Preferred option: use the repo utility added for this purpose.

```bash
cd ~/cs4248/project
python collect_valid_slurm_logs.py
```

This copies completed full-pipeline logs into:

```text
../candidate/logs/
```

and writes:

```text
../candidate/manifest.tsv
../candidate/manifest.jsonl
```

For a dry run:

```bash
python collect_valid_slurm_logs.py --dry-run
```

For resumed runs, loosen the filter:

```bash
python collect_valid_slurm_logs.py --allow-resume
```

For one model only:

```bash
python collect_valid_slurm_logs.py --model roberta-large
```

Manual one-off parser:

Run this from the cluster repo root:

```bash
cd ~/cs4248/project
python - <<'PY'
from pathlib import Path
import re

for log in sorted(Path("results").glob("slurm_new_*.log")):
    text = log.read_text(errors="ignore")
    job = re.search(r"Job ID:\s*(\d+)", text)
    node = re.search(r"Node:\s*(.+)", text)
    time = re.search(r"Time:\s*(.+)", text)
    run_line = re.search(r"Running phases from .* classifier=(\S+) model=(\S+)(?: include_d4=(\S+))?", text)
    model_line = re.search(r"Model:\s*(.*?)\s+\(--model\s+([^)]+)\)", text)
    include_d25 = re.search(r"Include D2\.5:\s*(\S+)", text)
    include_d4 = re.search(r"Include D4:\s*(\S+)", text)
    elapsed = re.search(r"All phases complete in\s*([\d.]+)\s*minutes", text)
    stopped = re.search(r"Pipeline stopped at phase\s*(\d+)", text)

    job_id = job.group(1) if job else log.stem.split("_")[-1]
    model = "?"
    classifier = "?"
    if run_line:
        classifier = run_line.group(1)
        model = run_line.group(2)
    if model_line:
        model = model_line.group(2)

    status = "completed" if elapsed else "failed"
    if stopped:
        status = f"failed_phase_{stopped.group(1)}"

    print(
        "\t".join([
            job_id,
            status,
            model,
            classifier,
            (include_d25.group(1) if include_d25 else "?"),
            (include_d4.group(1) if include_d4 else "?"),
            (elapsed.group(1) if elapsed else ""),
            (node.group(1).strip() if node else ""),
            (time.group(1).strip() if time else ""),
            str(log),
        ])
    )
PY
```

Suggested header for the output:

```text
job_id	status	model	classifier	include_d25	include_d4	elapsed_min	node	time	log_path
```

## Error-Log Parser

If some jobs failed before writing useful `.log` output, scan `.err` too:

```bash
cd ~/cs4248/project
python - <<'PY'
from pathlib import Path
import re

for err in sorted(Path("results").glob("slurm_new_*.err")):
    text = err.read_text(errors="ignore")
    job_id = err.stem.split("_")[-1]
    if "OutOfMemoryError" in text or "oom_kill" in text:
        reason = "oom"
    elif "AttributeError" in text:
        reason = "attribute_error"
    elif "Traceback" in text:
        reason = "traceback"
    elif text.strip():
        reason = "stderr_nonempty"
    else:
        reason = "stderr_empty"

    first_error = ""
    for line in text.splitlines():
        if "Traceback" in line or "OutOfMemoryError" in line or "AttributeError" in line or "oom_kill" in line:
            first_error = line.strip()
            break

    print("\t".join([job_id, reason, first_error, str(err)]))
PY
```

## Recommendation For Future Runs

Add a tiny metadata file per run so job IDs survive even if logs are moved. The cleanest place is inside the model cache:

```text
cache/<model_slug>/run_metadata.json
```

Suggested fields:
- `slurm_job_id`
- `run_model`
- `classifier`
- `include_d25`
- `include_d4`
- `start_phase`
- `only_phase`
- `node`
- `start_time`
- `git_commit`

For now, treat the Slurm logs as the source of truth.
