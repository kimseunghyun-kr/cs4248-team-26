"""
Shared cache-path helpers for the evaluation conditions.
"""

from __future__ import annotations

import os
from collections import OrderedDict


def _build_condition_specs() -> OrderedDict:
    specs = OrderedDict(
        [
            ("B1 (raw)", {"slug": "b1_raw"}),
            ("D1 (debias_vl)", {"slug": "d1_debias_vl"}),
            ("D2 (CBDC)", {"slug": "d2_cbdc"}),
        ]
    )
    if os.environ.get("INCLUDE_D25") == "1":
        specs["D2.5 (CBDC no-label-select)"] = {"slug": "d25_cbdc_no_label_select"}
    specs["D3 (debias_vl->CBDC)"] = {"slug": "d3_debias_vl_cbdc"}
    return specs


CONDITION_SPECS = _build_condition_specs()

SLUG_TO_LABEL = {spec["slug"]: label for label, spec in CONDITION_SPECS.items()}


def iter_condition_labels() -> list[str]:
    return list(CONDITION_SPECS.keys())


def _resolve_slug(label_or_slug: str) -> str:
    if label_or_slug in CONDITION_SPECS:
        return CONDITION_SPECS[label_or_slug]["slug"]
    if label_or_slug in SLUG_TO_LABEL:
        return label_or_slug
    raise KeyError(f"Unknown condition label/slug: {label_or_slug}")


def raw_split_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"z_tweet_{split}.pt")


def conditions_root(cache_dir: str) -> str:
    return os.path.join(cache_dir, "conditions")


def condition_dir(cache_dir: str, label_or_slug: str) -> str:
    return os.path.join(conditions_root(cache_dir), _resolve_slug(label_or_slug))


def ensure_condition_dir(cache_dir: str, label_or_slug: str) -> str:
    path = condition_dir(cache_dir, label_or_slug)
    os.makedirs(path, exist_ok=True)
    return path


def condition_split_path(cache_dir: str, label_or_slug: str, split: str) -> str:
    return os.path.join(condition_dir(cache_dir, label_or_slug), f"z_tweet_{split}.pt")


def condition_artifact_path(cache_dir: str, label_or_slug: str, filename: str) -> str:
    return os.path.join(condition_dir(cache_dir, label_or_slug), filename)
