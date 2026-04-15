"""
End-to-end orchestrator for the B1 / D1 / D2 / D3 pipeline (debug variant).
"""

import sys
import os
import argparse
import time
import subprocess
import importlib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_DIR)
from config import MODEL_REGISTRY, get_model_name, model_slug

def get_phases(classifier: str):
    phase3_module = "pipeline.classify"
    phase3_desc = "Supervised linear-probe evaluation"
    if classifier == "prototype":
        phase3_module = "pipeline.prototype_classify"
        phase3_desc = "Prototype-based evaluation"

    return [
        (1, "data.embed", "Embedding extraction"),
        (2, "cbdc.refine", "Materialize D1 / D2 / D2.5 / D3"),
        (3, phase3_module, phase3_desc),
        (4, "pipeline.evaluate", "Full evaluation report"),
    ]


def run_phase_subprocess(module_name: str, description: str, extra_env: dict) -> bool:
    script_path = os.path.join(PROJECT_DIR, module_name.replace(".", os.sep) + ".py")

    print(f"\n{'#'*70}")
    print(f"# PHASE: {description}")
    print(f"# Script: {script_path}")
    print(f"{'#'*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-u", script_path],
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONUNBUFFERED": "1", **extra_env},
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] Phase '{description}' failed (rc={result.returncode})")
        print("        Fix and re-run with --start_phase <N>")
        return False

    print(f"\n[OK] Phase '{description}' completed in {elapsed:.1f}s")
    return True


def run_phase_inprocess(module_name: str, description: str, extra_env: dict) -> bool:
    print(f"\n{'#'*70}")
    print(f"# PHASE: {description}")
    print(f"# Module: {module_name}")
    print(f"{'#'*70}\n")

    old_env = os.environ.copy()
    os.environ.update({"PYTHONUNBUFFERED": "1", **extra_env})

    t0 = time.time()
    try:
        module = importlib.import_module(module_name)

        if not hasattr(module, "main"):
            raise AttributeError(f"Module {module_name} does not define main()")

        module.main()
    except Exception as e:
        print(f"\n[ERROR] Phase '{description}' failed: {e}")
        return False
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    elapsed = time.time() - t0
    print(f"\n[OK] Phase '{description}' completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the B1 / D1 / D2 / D3 pipeline (debug).")
    parser.add_argument("--start_phase", type=int, default=1,
                        help="Resume from this phase (1-4).")
    parser.add_argument("--only_phase", type=int, default=None,
                        help="Run only this phase.")
    parser.add_argument("--model", default="bert",
                        help=f"Backbone encoder. Registry shortcuts: {list(MODEL_REGISTRY.keys())}. "
                             "Or pass any HuggingFace model ID.")
    parser.add_argument("--tokenizer", default=None,
                        help="Optional custom tokenizer (HuggingFace ID).")
    parser.add_argument("--text_unit", default="text",
                        help="Text unit for prompts: 'text', 'tweet', 'review', etc.")
    parser.add_argument("--classifier", default="probe", choices=["probe", "prototype"],
                        help="Phase-3 prediction method.")
    parser.add_argument("--skip_cbdc", action="store_true",
                        help="Skip Phase 2 materialization and run baseline-only evaluation.")
    parser.add_argument("--include_d25", action="store_true",
                        help="Also materialize D2.5 (CBDC with label-free checkpoint selection).")
    parser.add_argument("--inprocess", action="store_true",
                        help="Run phases in-process for PyCharm debugging.")
    args = parser.parse_args()

    hf_model_name = get_model_name(args.model)
    slug = model_slug(args.model)
    cache_dir = os.path.join(PROJECT_DIR, "cache", slug)
    extra_env = {
        "MODEL_NAME": hf_model_name,
        "CACHE_DIR": cache_dir,
        "TEXT_UNIT": args.text_unit,
    }
    if args.tokenizer:
        extra_env["TOKENIZER_NAME"] = args.tokenizer
    if args.include_d25:
        extra_env["INCLUDE_D25"] = "1"

    if args.classifier == "prototype":
        extra_env["RESULTS_FILE"] = "results_prototype.pt"
        extra_env["REPORT_FILE"] = "eval_report_prototype.txt"
        extra_env["EVAL_TITLE"] = "CBDC Sentiment Debiasing — Prototype Evaluation Report"
        extra_env["RESULTS_SECTION_TITLE"] = "Prototype Results"
    else:
        extra_env["RESULTS_FILE"] = "results.pt"
        extra_env["REPORT_FILE"] = "eval_report.txt"
        extra_env["EVAL_TITLE"] = "CBDC Sentiment Debiasing — Supervised Evaluation Report"
        extra_env["RESULTS_SECTION_TITLE"] = "Supervised Results"

    phases_all = get_phases(args.classifier)
    if args.only_phase is not None:
        phases = [p for p in phases_all if p[0] == args.only_phase]
        if not phases:
            print(f"ERROR: Phase {args.only_phase} not found. Valid: 1-{len(phases_all)}")
            sys.exit(1)
    else:
        phases = [p for p in phases_all if p[0] >= args.start_phase]

    if args.skip_cbdc:
        if args.classifier == "prototype":
            print("ERROR: --skip_cbdc cannot be used with --classifier prototype.")
            print("       Prototype evaluation needs Phase 2 to materialize class_prompt_prototypes.pt.")
            sys.exit(1)
        phases = [p for p in phases if p[0] != 2]

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "results"), exist_ok=True)

    print("=" * 70)
    print("CBDC Sentiment Debiasing Pipeline (debug)")
    print("=" * 70)
    print(f"Model:          {hf_model_name}  (--model {args.model})")
    if args.tokenizer:
        print(f"Tokenizer:      {args.tokenizer}")
    print(f"Text unit:      {args.text_unit}")
    print(f"Classifier:     {args.classifier}")
    print(f"Cache dir:      {cache_dir}")
    print(f"Running phases: {[p[0] for p in phases]}")
    print(f"Include D2.5:   {args.include_d25}")
    print(f"In-process:     {args.inprocess}")
    if args.skip_cbdc:
        print(f"Phase 2:        SKIPPED (baseline only)")

    total_start = time.time()

    for num, module_name, desc in phases:
        label = f"[{num}/{len(phases_all)}] {desc}"
        ok = (
            run_phase_inprocess(module_name, label, extra_env)
            if args.inprocess
            else run_phase_subprocess(module_name, label, extra_env)
        )
        if not ok:
            print(f"\nPipeline stopped at phase {num}.")
            print(f"To resume: python run_all.py --start_phase {num}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    report_name = extra_env["REPORT_FILE"]
    print(f"\n{'='*70}")
    print(f"All phases complete in {total_elapsed/60:.1f} minutes.")
    print(f"Results: {os.path.join(PROJECT_DIR, 'results', report_name)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
