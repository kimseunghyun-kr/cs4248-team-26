"""
Optional tweet-level sentiment feature utilities.

These features are intentionally opt-in so the default pipeline remains
unchanged. The current integration supports tweet-level VADER and AFINN
signals that can be concatenated to cached embeddings or transformer
representations.
"""

from __future__ import annotations

from functools import lru_cache

import torch


VADER_FEATURE_NAMES = [
    "vader_compound",
    "vader_pos",
    "vader_neg",
    "vader_neu",
]

AFINN_FEATURE_NAMES = [
    "afinn_total",
    "afinn_pos_count",
    "afinn_neg_count",
    "afinn_pos_total",
    "afinn_neg_total",
]


def _clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return str(text)


@lru_cache(maxsize=1)
def _get_vader_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "VADER features requested, but vaderSentiment is not installed. "
            "Install it with `pip install vaderSentiment afinn`."
        ) from exc
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _get_afinn_analyzer():
    try:
        from afinn import Afinn
    except ImportError as exc:
        raise RuntimeError(
            "AFINN features requested, but afinn is not installed. "
            "Install it with `pip install vaderSentiment afinn`."
        ) from exc
    return Afinn()


def compute_vader_features(text: str) -> list[float]:
    scores = _get_vader_analyzer().polarity_scores(_clean_text(text))
    return [
        float(scores["compound"]),
        float(scores["pos"]),
        float(scores["neg"]),
        float(scores["neu"]),
    ]


def compute_afinn_features(text: str) -> list[float]:
    afinn = _get_afinn_analyzer()
    tokens = _clean_text(text).split()
    word_scores = [float(afinn.score(token)) for token in tokens]
    total = float(sum(word_scores))
    pos_count = float(sum(1 for score in word_scores if score > 0))
    neg_count = float(sum(1 for score in word_scores if score < 0))
    pos_total = float(sum(score for score in word_scores if score > 0))
    neg_total = float(sum(score for score in word_scores if score < 0))
    return [total, pos_count, neg_count, pos_total, neg_total]


def get_requested_feature_names(
    use_vader_features: bool = False,
    use_afinn_features: bool = False,
) -> list[str]:
    names: list[str] = []
    if use_vader_features:
        names.extend(VADER_FEATURE_NAMES)
    if use_afinn_features:
        names.extend(AFINN_FEATURE_NAMES)
    return names


def compute_requested_text_features(
    text: str,
    use_vader_features: bool = False,
    use_afinn_features: bool = False,
) -> list[float]:
    features: list[float] = []
    if use_vader_features:
        features.extend(compute_vader_features(text))
    if use_afinn_features:
        features.extend(compute_afinn_features(text))
    return features


def build_text_feature_matrix(
    texts: list[str],
    use_vader_features: bool = False,
    use_afinn_features: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    feature_names = get_requested_feature_names(use_vader_features, use_afinn_features)
    if not feature_names:
        return None

    rows = [
        compute_requested_text_features(
            text,
            use_vader_features=use_vader_features,
            use_afinn_features=use_afinn_features,
        )
        for text in texts
    ]
    return torch.tensor(rows, dtype=dtype)
