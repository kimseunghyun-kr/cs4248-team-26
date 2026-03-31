"""
Concept prompts and word-pair definitions for CBDC financial sentiment.

Maps to original CBDC:
  cls_text      ↔  class concepts ("landbird"/"waterbird")
  test_text     ↔  neutral concepts for L_s ("bird", "celebrity")
  candidate_prompt ↔  debias_vl crossed grid (class × spurious)
  spurious_prompt  ↔  pure spurious descriptions ("water background")
  S_pairs       ↔  same class, different spurious (semantic preservation)
  B_pairs       ↔  same spurious, different class (bias direction)

Grid: 3 sentiments × 32 topics = 96 candidate prompts
  S_pairs: 3 × C(32,2) = 3 × 496 = 1488  (semantic preservation)
  B_pairs: 32 × 3 = 96                    (bias direction)
"""

import torch
import torch.nn.functional as F

# ── Sentiment class concepts ────────────────────────────────────────────
# Analogous to "landbird" / "waterbird" in Waterbirds
cls_text = [
    "This expresses negative financial sentiment.",
    "This expresses neutral financial sentiment.",
    "This expresses positive financial sentiment.",
]

# ── Neutral concepts for L_s preservation ───────────────────────────────
# Analogous to "bird", "celebrity" — must NOT correlate with sentiment
test_text = [
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
        cls_cb       (3, H)   sentiment class embeddings
        test_cb      (12, H)  neutral concept embeddings
        candidate_cb (96, H)  sentiment × topic crossed
        spurious_cb  (32, H)  pure topic embeddings
    """
    return {
        "cls_cb":       encoder.encode_text(cls_text),
        "test_cb":      encoder.encode_text(test_text),
        "candidate_cb": encoder.encode_text(candidate_prompt),
        "spurious_cb":  encoder.encode_text(spurious_prompt),
    }
