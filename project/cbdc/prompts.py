"""
Prompt banks for CBDC financial tweet sentiment.

The static prompt roles stay the same:
  cls_text_groups   ↔ class concepts pooled into sentiment prototypes
  target_text       ↔ prompts attacked by PGD and matched to class prototypes
  keep_text         ↔ neutral finance prompts used only for L_s

For the debias_vl stage, topics can now be mined from the tweet training split:
  candidate_prompt  ↔ sentiment × mined-topic crossed grid
  spurious_prompt   ↔ mined-topic prompts without sentiment
  S_pairs           ↔ same sentiment, different topic
  B_pairs           ↔ same topic, different sentiment

If mining fails or the cached tweet split is unavailable, the code falls back to
the original hand-written topic list so the pipeline remains runnable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
import re
from typing import Sequence

import torch
import torch.nn.functional as F


_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
DEFAULT_CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_USER_RE = re.compile(r"@\w+")
_SYMBOL_GAP_RE = re.compile(r"([#$])\s+([a-z0-9_]+)")
_TOKEN_RE = re.compile(r"\$[a-z][a-z0-9_]*|#[a-z][a-z0-9_]*|[a-z][a-z0-9']*")
_MAX_NGRAM = 3

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours",
    "rt", "she", "so", "that", "the", "their", "them", "they", "this", "to",
    "too", "us", "was", "we", "were", "will", "with", "you", "your",
}


# ── Sentiment class concepts (pooled prototypes) ───────────────────────
cls_text_groups = [
    [
        "A tweet expressing negative financial sentiment.",
        "This text conveys bearish or unfavorable market sentiment.",
        "The writer sounds negative about the company or asset.",
        "A pessimistic financial post.",
    ],
    [
        "A tweet expressing neutral financial sentiment.",
        "This text is informational and emotionally neutral about the market.",
        "The writer sounds balanced and non-committal about the company or asset.",
        "A factual financial post without strong sentiment.",
    ],
    [
        "A tweet expressing positive financial sentiment.",
        "This text conveys bullish or favorable market sentiment.",
        "The writer sounds positive about the company or asset.",
        "An optimistic financial post.",
    ],
]

cls_group_sizes = [len(group) for group in cls_text_groups]
cls_text = [group[0] for group in cls_text_groups]


# ── PGD targets for RN50-style text_iccv matching ──────────────────────
target_text = [
    "A negative-sentiment financial tweet.",
    "A neutral-sentiment financial tweet.",
    "A positive-sentiment financial tweet.",
]


# ── Neutral finance prompts for L_s preservation ────────────────────────
keep_text = [
    "A discussion about market conditions.",
    "Commentary on recent financial developments.",
    "An observation about trading activity.",
    "A remark about economic indicators.",
    "Thoughts on investment performance.",
    "A perspective on market trends.",
    "A note about financial news coverage.",
    "An analysis of sector movements.",
    "A comment on portfolio allocation.",
    "A report on capital market dynamics.",
    "An overview of global economic conditions.",
    "A summary of recent corporate disclosures.",
]


# ── Static fallback topics ──────────────────────────────────────────────
DEFAULT_TOPICS = [
    "cryptocurrency",
    "equities",
    "bonds",
    "commodities",
    "real estate",
    "derivatives",
    "the technology sector",
    "the energy sector",
    "the healthcare sector",
    "the financial sector",
    "consumer goods and retail",
    "industrials and manufacturing",
    "telecommunications",
    "utilities",
    "interest rates",
    "inflation",
    "economic growth",
    "the labor market",
    "trade policy",
    "central bank policy",
    "government fiscal policy",
    "exchange rates",
    "corporate earnings",
    "mergers and acquisitions",
    "initial public offerings",
    "dividend announcements",
    "regulatory actions",
    "credit ratings",
    "US markets",
    "emerging markets",
    "European markets",
    "Asian markets",
]


def _normalize_tweet_text(text: str) -> str:
    text = text.lower()
    text = text.replace("&amp;", " and ")
    text = _URL_RE.sub(" ", text)
    text = _USER_RE.sub(" ", text)
    text = _SYMBOL_GAP_RE.sub(r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_tweet_text(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize_tweet_text(text))


def _is_valid_phrase(tokens: Sequence[str]) -> bool:
    if not tokens or len(tokens) > _MAX_NGRAM:
        return False
    if all(tok in _STOPWORDS for tok in tokens):
        return False
    if len(tokens) > 1 and (tokens[0] in _STOPWORDS or tokens[-1] in _STOPWORDS):
        return False
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in _STOPWORDS:
            return False
        if len(tok) < 3 and not tok.startswith(("$", "#")):
            return False
    return True


def _iter_phrase_candidates(tokens: Sequence[str]) -> set[str]:
    phrases = set()
    for n in range(1, _MAX_NGRAM + 1):
        for i in range(0, len(tokens) - n + 1):
            chunk = tokens[i : i + n]
            if not _is_valid_phrase(chunk):
                continue
            phrases.add(" ".join(chunk))
    return phrases


def _is_redundant_phrase(phrase: str, chosen: Sequence[str]) -> bool:
    phrase_tokens = set(phrase.split())
    for other in chosen:
        if phrase == other or phrase in other or other in phrase:
            return True
        other_tokens = set(other.split())
        union = phrase_tokens | other_tokens
        if union and len(phrase_tokens & other_tokens) / len(union) >= 0.8:
            return True
    return False


def _label_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log(p)
    return entropy / math.log(3.0)


def mine_topic_phrases_from_texts(
    texts: Sequence[str],
    labels: Sequence[int],
    max_topics: int = 32,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
) -> list[dict]:
    """Mine label-balanced topic/style phrases from tweet texts.

    Phrases are ranked by:
      frequency × class-balance entropy × small specificity bonus
    and then greedily deduplicated for diversity.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    doc_counts = Counter()
    label_doc_counts = defaultdict(lambda: [0, 0, 0])
    n_docs = len(texts)
    max_doc_freq = max(min_doc_freq + 1, int(max_doc_freq_ratio * n_docs))

    for text, label in zip(texts, labels):
        phrases = _iter_phrase_candidates(_tokenize_tweet_text(text))
        for phrase in phrases:
            doc_counts[phrase] += 1
            label_doc_counts[phrase][int(label)] += 1

    scored = []
    for phrase, total in doc_counts.items():
        if total < min_doc_freq or total > max_doc_freq:
            continue

        counts = label_doc_counts[phrase]
        entropy = _label_entropy(counts)
        if entropy < 0.45:
            continue

        ngram_len = phrase.count(" ") + 1
        has_symbol = int("$" in phrase or "#" in phrase)
        specificity_bonus = 1.0 + 0.10 * (ngram_len - 1) + 0.10 * has_symbol
        score = math.log1p(total) * entropy * specificity_bonus

        scored.append(
            {
                "phrase": phrase,
                "doc_freq": int(total),
                "label_counts": [int(x) for x in counts],
                "entropy": float(entropy),
                "score": float(score),
            }
        )

    scored.sort(key=lambda row: (row["score"], row["doc_freq"]), reverse=True)

    chosen = []
    chosen_phrases = []
    for row in scored:
        if _is_redundant_phrase(row["phrase"], chosen_phrases):
            continue
        chosen.append(row)
        chosen_phrases.append(row["phrase"])
        if len(chosen) >= max_topics:
            break

    return chosen


def _load_cached_texts(tokenizer, train_data: dict) -> list[str]:
    texts = train_data.get("texts")
    if texts is not None:
        return list(texts)
    if tokenizer is None:
        raise ValueError("Tokenizer is required to decode cached train split without raw texts.")
    return tokenizer.batch_decode(train_data["input_ids"], skip_special_tokens=True)


def mine_topic_phrases_from_cache(
    tokenizer,
    cache_dir: str | None = None,
    max_topics: int = 32,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
    force_refresh: bool = False,
) -> tuple[list[str], list[dict], bool]:
    """Mine topic phrases from cached train tweets and persist the result."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "mined_topics.json")

    if os.path.exists(out_path) and not force_refresh:
        with open(out_path, "r") as f:
            payload = json.load(f)
        topics = payload.get("topics", [])
        metadata = payload.get("metadata", [])
        using_mined = bool(payload.get("using_mined_topics", False))
        if topics:
            return topics[:max_topics], metadata[:max_topics], using_mined

    train_path = os.path.join(cache_dir, "z_tweet_train.pt")
    if not os.path.exists(train_path):
        topics = DEFAULT_TOPICS[:max_topics]
        metadata = []
        with open(out_path, "w") as f:
            json.dump(
                {
                    "using_mined_topics": False,
                    "topics": topics,
                    "metadata": metadata,
                    "reason": f"missing {train_path}",
                },
                f,
                indent=2,
            )
        return topics, metadata, False

    train_data = torch.load(train_path, map_location="cpu")
    texts = _load_cached_texts(tokenizer, train_data)
    labels = train_data["labels"].tolist()
    metadata = mine_topic_phrases_from_texts(
        texts,
        labels,
        max_topics=max_topics,
        min_doc_freq=min_doc_freq,
        max_doc_freq_ratio=max_doc_freq_ratio,
    )
    topics = [row["phrase"] for row in metadata]

    using_mined_topics = len(topics) >= min(8, max_topics)
    if not using_mined_topics:
        topics = DEFAULT_TOPICS[:max_topics]
        metadata = []

    with open(out_path, "w") as f:
        json.dump(
            {
                "using_mined_topics": using_mined_topics,
                "topics": topics,
                "metadata": metadata,
            },
            f,
            indent=2,
        )

    return topics, metadata, using_mined_topics


def build_prompt_bank(
    topics: Sequence[str],
    topic_metadata: Sequence[dict] | None = None,
    using_mined_topics: bool = False,
) -> dict:
    """Build the debias_vl prompt grid and index pairs for a topic list."""
    topics = list(topics)
    n_topics = len(topics)
    candidate_prompt = (
        [f"A negative tweet about {topic}." for topic in topics]
        + [f"A neutral tweet about {topic}." for topic in topics]
        + [f"A positive tweet about {topic}." for topic in topics]
    )
    spurious_prompt = [f"A tweet about {topic}." for topic in topics]

    S_pairs = []
    for offset in [0, n_topics, 2 * n_topics]:
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                S_pairs.append([offset + i, offset + j])

    B_pairs = []
    for topic_idx in range(n_topics):
        for s1, s2 in [(0, n_topics), (0, 2 * n_topics), (n_topics, 2 * n_topics)]:
            B_pairs.append([s1 + topic_idx, s2 + topic_idx])

    return {
        "topics": topics,
        "topic_metadata": list(topic_metadata or []),
        "using_mined_topics": using_mined_topics,
        "cls_text_groups": cls_text_groups,
        "cls_group_sizes": cls_group_sizes,
        "cls_text": cls_text,
        "target_text": target_text,
        "keep_text": keep_text,
        "candidate_prompt": candidate_prompt,
        "spurious_prompt": spurious_prompt,
        "S_pairs": S_pairs,
        "B_pairs": B_pairs,
    }


def get_prompt_bank(
    tokenizer=None,
    cache_dir: str | None = None,
    use_mined_topics: bool = True,
    max_topics: int = 32,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
    force_refresh: bool = False,
) -> dict:
    """Return the active prompt bank, preferring mined tweet phrases."""
    if use_mined_topics:
        try:
            topics, metadata, using_mined = mine_topic_phrases_from_cache(
                tokenizer=tokenizer,
                cache_dir=cache_dir,
                max_topics=max_topics,
                min_doc_freq=min_doc_freq,
                max_doc_freq_ratio=max_doc_freq_ratio,
                force_refresh=force_refresh,
            )
            return build_prompt_bank(
                topics,
                topic_metadata=metadata,
                using_mined_topics=using_mined,
            )
        except Exception as exc:
            bank = build_prompt_bank(DEFAULT_TOPICS[:max_topics], using_mined_topics=False)
            bank["mining_error"] = str(exc)
            return bank

    return build_prompt_bank(DEFAULT_TOPICS[:max_topics], using_mined_topics=False)


DEFAULT_PROMPT_BANK = build_prompt_bank(DEFAULT_TOPICS, using_mined_topics=False)
candidate_prompt = DEFAULT_PROMPT_BANK["candidate_prompt"]
spurious_prompt = DEFAULT_PROMPT_BANK["spurious_prompt"]
S_pairs = DEFAULT_PROMPT_BANK["S_pairs"]
B_pairs = DEFAULT_PROMPT_BANK["B_pairs"]


def encode_all_prompts(encoder, prompt_bank: dict | None = None) -> dict:
    """Encode every prompt set with the given encoder."""
    bank = prompt_bank or DEFAULT_PROMPT_BANK
    return {
        "cls_cb":       encode_grouped_prompts(encoder, bank["cls_text_groups"]),
        "target_cb":    encoder.encode_text(bank["target_text"]),
        "keep_cb":      encoder.encode_text(bank["keep_text"]),
        "candidate_cb": encoder.encode_text(bank["candidate_prompt"]),
        "spurious_cb":  encoder.encode_text(bank["spurious_prompt"]),
    }


def flatten_prompt_groups(prompt_groups: Sequence[Sequence[str]]) -> list[str]:
    """Flatten prompt groups while preserving class order."""
    return [prompt for group in prompt_groups for prompt in group]


def pool_prompt_group_embeddings(
    embeddings: torch.Tensor,
    group_sizes: Sequence[int],
    normalize: bool = True,
) -> torch.Tensor:
    """Average sequential chunks of embeddings into one vector per group."""
    pooled = []
    start = 0
    for size in group_sizes:
        chunk = embeddings[start : start + size]
        group_emb = chunk.mean(dim=0)
        if normalize:
            group_emb = F.normalize(group_emb, dim=-1)
        pooled.append(group_emb)
        start += size
    return torch.stack(pooled, dim=0)


def encode_grouped_prompts(encoder, prompt_groups: Sequence[Sequence[str]]) -> torch.Tensor:
    """Encode prompt groups and return one pooled embedding per group."""
    flat = flatten_prompt_groups(prompt_groups)
    sizes = [len(group) for group in prompt_groups]
    encoded = encoder.encode_text(flat)
    return pool_prompt_group_embeddings(encoded, sizes, normalize=True)
