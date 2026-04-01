"""
Shared cache-path helpers for the four official evaluation conditions.
"""

from __future__ import annotations

import os
from collections import OrderedDict


CONDITION_SPECS = OrderedDict(
    [
        ("B1 (raw)", {"slug": "b1_raw"}),
        ("D1 (debias_vl)", {"slug": "d1_debias_vl"}),
        ("D2 (CBDC)", {"slug": "d2_cbdc"}),
        ("D3 (debias_vl->CBDC)", {"slug": "d3_debias_vl_cbdc"}),
    ]
)

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
