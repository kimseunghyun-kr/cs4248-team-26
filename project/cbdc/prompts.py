"""
Concept prompts and word-pair definitions for CBDC financial sentiment.

Prompt roles are split to match the RN50 `text_iccv` pathway more closely:
  cls_text_groups   ↔ class concepts pooled into sentiment prototypes
  target_text       ↔ prompts attacked by PGD and matched to class prototypes
  keep_text         ↔ neutral finance prompts used only for L_s
  candidate_prompt  ↔ debias_vl crossed grid (sentiment × topic)
  spurious_prompt   ↔ pure topic descriptions
  S_pairs           ↔ same sentiment, different topic (semantic preservation)
  B_pairs           ↔ same topic, different sentiment (bias contrast pairs)

Grid: 3 sentiments × 32 topics = 96 candidate prompts
  S_pairs: 3 × C(32,2) = 1488
  B_pairs: 32 × 3 = 96
"""

from typing import Sequence

import torch
import torch.nn.functional as F

# ── Sentiment class concepts (pooled prototypes) ───────────────────────
# Multiple paraphrases make the class embeddings less brittle than a single
# template while still collapsing to three sentiment prototypes.
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

# Canonical one-line representatives kept for logging / downstream reporting.
cls_text = [group[0] for group in cls_text_groups]

# ── PGD targets for RN50-style text_iccv matching ──────────────────────
# These are the prompts whose latent representations are adversarially
# perturbed, producing S[i::3] that aligns to cls_em[i].
target_text = [
    "A negative-sentiment financial tweet.",
    "A neutral-sentiment financial tweet.",
    "A positive-sentiment financial tweet.",
]

# ── Neutral finance prompts for L_s preservation ────────────────────────
# These should carry finance semantics but be as sentiment-agnostic as possible.
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

# ── 32 financial topics (spurious attributes) ──────────────────────────
# Asset classes (6), Market sectors (8), Macroeconomic themes (8),
# Corporate events (6), Geographic scope (4)
_TOPICS = [
    # Asset classes
    "cryptocurrency",
    "equities",
    "bonds",
    "commodities",
    "real estate",
    "derivatives",
    # Market sectors
    "the technology sector",
    "the energy sector",
    "the healthcare sector",
    "the financial sector",
    "consumer goods and retail",
    "industrials and manufacturing",
    "telecommunications",
    "utilities",
    # Macroeconomic themes
    "interest rates",
    "inflation",
    "economic growth",
    "the labor market",
    "trade policy",
    "central bank policy",
    "government fiscal policy",
    "exchange rates",
    # Corporate events
    "corporate earnings",
    "mergers and acquisitions",
    "initial public offerings",
    "dividend announcements",
    "regulatory actions",
    "credit ratings",
    # Geographic scope
    "US markets",
    "emerging markets",
    "European markets",
    "Asian markets",
]

N_TOPICS = len(_TOPICS)  # 32

# ── debias_vl word-pair grid: sentiment × topic ────────────────────────
# 3 sentiments × 32 topics = 96 crossed prompts
candidate_prompt = (
    # Negative × 32 topics  (indices 0 – 31)
    [f"A negative view on {t}." for t in _TOPICS]
    # Neutral × 32 topics   (indices 32 – 63)
    + [f"A neutral observation about {t}." for t in _TOPICS]
    # Positive × 32 topics  (indices 64 – 95)
    + [f"A positive view on {t}." for t in _TOPICS]
)

# Pure topic descriptions (no sentiment)  — 32 entries
spurious_prompt = [f"A statement about {t}." for t in _TOPICS]

# ── Pair construction ───────────────────────────────────────────────────

# S_pairs: same sentiment, different topic → PRESERVE sentiment
# 3 × C(32, 2) = 3 × 496 = 1488 total
S_pairs = []
for _offset in [0, N_TOPICS, 2 * N_TOPICS]:
    for _i in range(N_TOPICS):
        for _j in range(_i + 1, N_TOPICS):
            S_pairs.append([_offset + _i, _offset + _j])

# B_pairs: same topic, different sentiment → BIAS to remove
# 32 × 3 = 96 total
B_pairs = []
for _topic in range(N_TOPICS):
    for _s1, _s2 in [(0, N_TOPICS), (0, 2 * N_TOPICS), (N_TOPICS, 2 * N_TOPICS)]:
        B_pairs.append([_s1 + _topic, _s2 + _topic])


def encode_all_prompts(encoder) -> dict:
    """Encode every prompt set with the given encoder.

    Returns dict with keys:
        cls_cb       (3, H)   pooled sentiment class embeddings
        target_cb    (3, H)   PGD target prompt embeddings
        keep_cb      (12, H)  neutral finance embeddings for L_s
        candidate_cb (96, H)  sentiment × topic crossed
        spurious_cb  (32, H)  pure topic embeddings
    """
    return {
        "cls_cb":       encode_grouped_prompts(encoder, cls_text_groups),
        "target_cb":    encoder.encode_text(target_text),
        "keep_cb":      encoder.encode_text(keep_text),
        "candidate_cb": encoder.encode_text(candidate_prompt),
        "spurious_cb":  encoder.encode_text(spurious_prompt),
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
