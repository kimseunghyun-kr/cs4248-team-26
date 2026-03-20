"""
End-to-end orchestrator for the SAE + CBDC financial sentiment pipeline.

Runs each phase in sequence by calling sub-scripts as subprocesses.
All phases are independent and can be run individually if a step fails.

Execution order:
  1. data/embed.py       — encode all corpora, cache to cache/
  2. sae/sae.py          — train sparse autoencoder
  3. sae/sae_analysis.py — extract v_style and v_shift
  4. cbdc/refine.py      — CBDC-PGD: refine v_style → delta_star
  5. pipeline/clean.py   — project out all directions (all conditions)
  6. pipeline/classify.py — train linear probes for all conditions
  7. pipeline/evaluate.py — full evaluation report

Usage (from project/ directory):
  python run_all.py                          # run all phases
  python run_all.py --start_phase 4         # resume from phase 4
  python run_all.py --only_phase 7          # run only evaluation
"""

import subprocess
import sys
import os
import argparse
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PHASES = [
    (1, "data/embed.py",          "Embedding extraction"),
    (2, "sae/sae.py",             "SAE training"),
    (3, "sae/sae_analysis.py",    "Style direction extraction"),
    (4, "cbdc/refine.py",         "CBDC-PGD refinement"),
    (5, "pipeline/clean.py",      "Orthogonal projection (all conditions)"),
    (6, "pipeline/classify.py",   "Linear probe training + eval"),
    (7, "pipeline/evaluate.py",   "Full evaluation report"),
]


def run_phase(script_path: str, description: str) -> bool:
    """Run a phase script. Returns True on success, False on failure."""
    abs_path = os.path.join(PROJECT_DIR, script_path)
    print(f"\n{'#'*70}")
    print(f"# PHASE: {description}")
    print(f"# Script: {script_path}")
    print(f"{'#'*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-u", abs_path],
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] Phase '{description}' failed with return code {result.returncode}")
        print(f"        Fix the error above and re-run with --start_phase <N>")
        return False

    print(f"\n[OK] Phase '{description}' completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full SAE+CBDC sentiment pipeline.")
    parser.add_argument("--start_phase", type=int, default=1,
                        help="Resume from this phase number (1-7). Default=1 (run all).")
    parser.add_argument("--only_phase",  type=int, default=None,
                        help="Run only this phase number (overrides start_phase).")
    args = parser.parse_args()

    if args.only_phase is not None:
        phases_to_run = [p for p in PHASES if p[0] == args.only_phase]
        if not phases_to_run:
            print(f"ERROR: Phase {args.only_phase} not found. Valid: 1-{len(PHASES)}")
            sys.exit(1)
    else:
        phases_to_run = [p for p in PHASES if p[0] >= args.start_phase]

    os.makedirs(os.path.join(PROJECT_DIR, "cache"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "results"), exist_ok=True)

    print("=" * 70)
    print("SAE + CBDC Financial Sentiment Pipeline")
    print("=" * 70)
    print(f"Running phases: {[p[0] for p in phases_to_run]}")
    print(f"Project dir: {PROJECT_DIR}")

    total_start = time.time()
    for phase_num, script, description in phases_to_run:
        success = run_phase(script, f"[{phase_num}/{len(PHASES)}] {description}")
        if not success:
            print(f"\nPipeline stopped at phase {phase_num}.")
            print(f"To resume: python run_all.py --start_phase {phase_num}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All phases complete in {total_elapsed/60:.1f} minutes.")
    print(f"Results: {os.path.join(PROJECT_DIR, 'results', 'eval_report_new.txt')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
