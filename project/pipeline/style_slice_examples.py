"""
Style-slice prototype analysis for B1/D1/D2/... conditions.

This script answers questions like:
  "After debiasing, are emoticon-heavy tweets classified differently?"

It recomputes per-example prototype predictions for cached condition embeddings,
then filters the fixed test split by surface-style cues such as emoticons,
repeated punctuation, laughter words, and very short text.

Run from project/ on the cluster after Phase 2/3 artifacts exist, for example:

  CACHE_DIR=cache/roberta INCLUDE_D25=1 INCLUDE_D4=1 \
    python pipeline/style_slice_examples.py --cache_dir cache/roberta
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_NAMES = ["negative", "neutral", "positive"]
B1_LABEL = "B1 (raw)"


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    slug: str


KNOWN_CONDITIONS = [
    ConditionSpec("B1 (raw)", "b1_raw"),
    ConditionSpec("D1 (debias_vl)", "d1_debias_vl"),
    ConditionSpec("D2 (CBDC)", "d2_cbdc"),
    ConditionSpec("D2.5 (CBDC no-label-select)", "d25_cbdc_no_label_select"),
    ConditionSpec("D3 (debias_vl->CBDC)", "d3_debias_vl_cbdc"),
    ConditionSpec("D4 (adv-discovery->CBDC)", "d4_adv_discovery_cbdc"),
]

SLUG_TO_LABEL = {spec.slug: spec.label for spec in KNOWN_CONDITIONS}
LABEL_TO_SLUG = {spec.label: spec.slug for spec in KNOWN_CONDITIONS}


ASCII_EMOTICON_RE = re.compile(
    r"(?i)(:-?\)|:-?d|;-?\)|:-?\(|:'\(|:-?p|x-d|xd|<3|\^_\^)"
)
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]"
)
LAUGHTER_RE = re.compile(r"(?i)\b(lol|lmao|lmfao|rofl|haha+|hehe+)\b")
URL_RE = re.compile(r"(?i)(https?://|www\.|bit\.ly|t\.co/)")
ALL_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def _has_emoticon_or_emoji(text: str) -> bool:
    return bool(ASCII_EMOTICON_RE.search(text) or EMOJI_RE.search(text))


def _token_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _style_detectors() -> dict[str, Callable[[str], bool]]:
    base_detectors: dict[str, Callable[[str], bool]] = {
        "emoticon_or_emoji": _has_emoticon_or_emoji,
        "exclamation_heavy": lambda text: bool(re.search(r"!{2,}", text)),
        "question_heavy": lambda text: bool(re.search(r"\?{2,}", text)),
        "laughter_word": lambda text: bool(LAUGHTER_RE.search(text)),
        "very_short": lambda text: _token_count(text) <= 4,
        "all_caps_token": lambda text: bool(ALL_CAPS_RE.search(text)),
        "url_or_link": lambda text: bool(URL_RE.search(text)),
    }

    def any_prompt_style(text: str) -> bool:
        return any(detector(text) for detector in base_detectors.values())

    def plain_no_prompt_style(text: str) -> bool:
        return not any_prompt_style(text)

    return {
        **base_detectors,
        "any_prompt_style": any_prompt_style,
        "plain_no_prompt_style": plain_no_prompt_style,
    }


def _split_path(cache_dir: str, condition_label: str, split: str) -> str:
    if condition_label == B1_LABEL:
        return os.path.join(cache_dir, f"z_tweet_{split}.pt")
    return os.path.join(
        cache_dir,
        "conditions",
        LABEL_TO_SLUG[condition_label],
        f"z_tweet_{split}.pt",
    )


def _load_payload(cache_dir: str, condition_label: str, split: str) -> dict | None:
    path = _split_path(cache_dir, condition_label, split)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def _load_prototypes(cache_dir: str, condition_label: str) -> torch.Tensor | None:
    path = os.path.join(
        cache_dir,
        "conditions",
        LABEL_TO_SLUG[condition_label],
        "class_prompt_prototypes.pt",
    )
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu").float()


@torch.no_grad()
def _prototype_logits(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z.float(), dim=-1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    return z @ prototypes.T


def _prediction_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def _condition_label(value: str) -> str:
    value = value.strip()
    if value in LABEL_TO_SLUG:
        return value
    if value in SLUG_TO_LABEL:
        return SLUG_TO_LABEL[value]
    raise KeyError(f"Unknown condition '{value}'. Use one of: {', '.join(LABEL_TO_SLUG)}")


def _available_conditions(cache_dir: str, requested: list[str] | None) -> list[str]:
    if requested:
        candidates = [_condition_label(item) for item in requested]
    else:
        candidates = [spec.label for spec in KNOWN_CONDITIONS]

    available = []
    for label in candidates:
        if _load_payload(cache_dir, label, "test") is None:
            continue
        if _load_prototypes(cache_dir, label) is None:
            continue
        available.append(label)
    return available


def _safe_f1(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return float("nan")
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def _safe_acc(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return float("nan")
    return float(accuracy_score(labels, preds))


def _counts(values: list[int]) -> dict[str, int]:
    return {f"{name}_count": int(sum(v == idx for v in values)) for idx, name in enumerate(LABEL_NAMES)}


def _fmt_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _write_metrics(
    out_path: str,
    rows: list[dict],
) -> None:
    fieldnames = [
        "style",
        "condition",
        "n",
        "accuracy",
        "macro_f1",
        "changed_vs_b1",
        "changed_rate_vs_b1",
        "corrected_vs_b1",
        "harmed_vs_b1",
        "b1_correct_condition_correct",
        "b1_wrong_condition_wrong",
        "true_negative_count",
        "true_neutral_count",
        "true_positive_count",
        "pred_negative_count",
        "pred_neutral_count",
        "pred_positive_count",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_examples(out_path: str, rows: list[dict]) -> None:
    fieldnames = [
        "style",
        "condition",
        "bucket",
        "text",
        "selected_text",
        "true_label",
        "b1_pred",
        "condition_pred",
        "b1_margin",
        "condition_margin",
        "b1_negative_score",
        "b1_neutral_score",
        "b1_positive_score",
        "condition_negative_score",
        "condition_neutral_score",
        "condition_positive_score",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bucket(b1_pred: int, cond_pred: int, gold: int) -> str | None:
    if b1_pred == cond_pred:
        return None
    b1_correct = b1_pred == gold
    cond_correct = cond_pred == gold
    if not b1_correct and cond_correct:
        return "corrected"
    if b1_correct and not cond_correct:
        return "harmed"
    return "changed_both_wrong"


def _pick_examples(
    *,
    style_name: str,
    condition_label: str,
    indices: list[int],
    labels: list[int],
    texts: list[str],
    selected_texts: list[str | None],
    b1_preds: list[int],
    b1_logits: torch.Tensor,
    b1_margins: torch.Tensor,
    cond_preds: list[int],
    cond_logits: torch.Tensor,
    cond_margins: torch.Tensor,
    max_examples_per_bucket: int,
) -> list[dict]:
    picked: list[dict] = []
    bucket_counts = {"corrected": 0, "harmed": 0, "changed_both_wrong": 0}

    for idx in indices:
        bucket = _bucket(b1_preds[idx], cond_preds[idx], labels[idx])
        if bucket is None:
            continue
        if bucket_counts[bucket] >= max_examples_per_bucket:
            continue
        bucket_counts[bucket] += 1
        picked.append(
            {
                "style": style_name,
                "condition": condition_label,
                "bucket": bucket,
                "text": texts[idx],
                "selected_text": selected_texts[idx] or "",
                "true_label": LABEL_NAMES[labels[idx]],
                "b1_pred": LABEL_NAMES[b1_preds[idx]],
                "condition_pred": LABEL_NAMES[cond_preds[idx]],
                "b1_margin": _fmt_float(float(b1_margins[idx])),
                "condition_margin": _fmt_float(float(cond_margins[idx])),
                "b1_negative_score": _fmt_float(float(b1_logits[idx, 0])),
                "b1_neutral_score": _fmt_float(float(b1_logits[idx, 1])),
                "b1_positive_score": _fmt_float(float(b1_logits[idx, 2])),
                "condition_negative_score": _fmt_float(float(cond_logits[idx, 0])),
                "condition_neutral_score": _fmt_float(float(cond_logits[idx, 1])),
                "condition_positive_score": _fmt_float(float(cond_logits[idx, 2])),
            }
        )

    return picked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create style-slice metrics and concrete changed-prediction examples."
    )
    parser.add_argument(
        "--cache_dir",
        default=os.environ.get("CACHE_DIR", "cache"),
        help="Model cache directory containing z_tweet_*.pt and conditions/* artifacts.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to <cache_dir>/style_slice_analysis.",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated labels/slugs to compare against B1. Defaults to all available.",
    )
    parser.add_argument(
        "--max_examples_per_bucket",
        type=int,
        default=12,
        help="Per style/condition examples for corrected, harmed, and changed_both_wrong.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    out_dir = args.out_dir or os.path.join(cache_dir, "style_slice_analysis")
    os.makedirs(out_dir, exist_ok=True)

    requested = None
    if args.conditions:
        requested = [item.strip() for item in args.conditions.split(",") if item.strip()]
    condition_labels = _available_conditions(cache_dir, requested)
    if B1_LABEL not in condition_labels:
        raise SystemExit(
            f"B1 artifacts are required. Missing test split or prototypes under: {cache_dir}"
        )

    b1_payload = _load_payload(cache_dir, B1_LABEL, "test")
    assert b1_payload is not None
    labels = b1_payload["labels"].long().cpu().tolist()
    texts = b1_payload.get("texts")
    if texts is None:
        raise SystemExit("Cached test split does not contain raw texts; rerun data/embed.py.")
    selected_texts = b1_payload.get("selected_texts") or [""] * len(texts)

    predictions: dict[str, dict[str, object]] = {}
    for condition_label in condition_labels:
        payload = _load_payload(cache_dir, condition_label, "test")
        prototypes = _load_prototypes(cache_dir, condition_label)
        if payload is None or prototypes is None:
            continue
        logits = _prototype_logits(payload["embeddings"], prototypes)
        preds = logits.argmax(dim=-1).cpu().tolist()
        predictions[condition_label] = {
            "logits": logits.cpu(),
            "preds": preds,
            "margins": _prediction_margin(logits).cpu(),
        }

    b1_preds = predictions[B1_LABEL]["preds"]
    b1_logits = predictions[B1_LABEL]["logits"]
    b1_margins = predictions[B1_LABEL]["margins"]

    detectors = _style_detectors()
    metric_rows: list[dict] = []
    example_rows: list[dict] = []

    for style_name, detector in detectors.items():
        indices = [idx for idx, text in enumerate(texts) if detector(text)]
        if not indices:
            continue

        style_labels = [labels[idx] for idx in indices]
        true_counts = {f"true_{k}": v for k, v in _counts(style_labels).items()}

        for condition_label, pred_payload in predictions.items():
            cond_preds = pred_payload["preds"]
            cond_logits = pred_payload["logits"]
            cond_margins = pred_payload["margins"]
            style_preds = [cond_preds[idx] for idx in indices]
            pred_counts = {f"pred_{k}": v for k, v in _counts(style_preds).items()}

            changed = 0
            corrected = 0
            harmed = 0
            both_correct = 0
            both_wrong = 0
            for idx in indices:
                b1_pred = b1_preds[idx]
                cond_pred = cond_preds[idx]
                gold = labels[idx]
                if b1_pred != cond_pred:
                    changed += 1
                if b1_pred != gold and cond_pred == gold:
                    corrected += 1
                if b1_pred == gold and cond_pred != gold:
                    harmed += 1
                if b1_pred == gold and cond_pred == gold:
                    both_correct += 1
                if b1_pred != gold and cond_pred != gold:
                    both_wrong += 1

            metric_rows.append(
                {
                    "style": style_name,
                    "condition": condition_label,
                    "n": len(indices),
                    "accuracy": _fmt_float(_safe_acc(style_labels, style_preds)),
                    "macro_f1": _fmt_float(_safe_f1(style_labels, style_preds)),
                    "changed_vs_b1": changed,
                    "changed_rate_vs_b1": _fmt_float(changed / len(indices)),
                    "corrected_vs_b1": corrected,
                    "harmed_vs_b1": harmed,
                    "b1_correct_condition_correct": both_correct,
                    "b1_wrong_condition_wrong": both_wrong,
                    **true_counts,
                    **pred_counts,
                }
            )

            if condition_label == B1_LABEL:
                continue
            example_rows.extend(
                _pick_examples(
                    style_name=style_name,
                    condition_label=condition_label,
                    indices=indices,
                    labels=labels,
                    texts=texts,
                    selected_texts=selected_texts,
                    b1_preds=b1_preds,
                    b1_logits=b1_logits,
                    b1_margins=b1_margins,
                    cond_preds=cond_preds,
                    cond_logits=cond_logits,
                    cond_margins=cond_margins,
                    max_examples_per_bucket=args.max_examples_per_bucket,
                )
            )

    metrics_path = os.path.join(out_dir, "style_slice_metrics.csv")
    examples_path = os.path.join(out_dir, "style_slice_examples.csv")
    _write_metrics(metrics_path, metric_rows)
    _write_examples(examples_path, example_rows)

    print(f"Conditions: {', '.join(predictions)}")
    print(f"Saved metrics  -> {metrics_path}")
    print(f"Saved examples -> {examples_path}")
    if "emoticon_or_emoji" in {row["style"] for row in metric_rows}:
        print("Tip: filter style_slice_examples.csv where style=emoticon_or_emoji.")


if __name__ == "__main__":
    main()
