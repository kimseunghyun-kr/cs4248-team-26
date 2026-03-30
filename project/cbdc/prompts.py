"""
Concept prompts and word-pair definitions for CBDC financial sentiment.

Maps to original CBDC:
  cls_text      ↔  class concepts ("landbird"/"waterbird")
  test_text     ↔  neutral concepts for L_s ("bird", "celebrity")
  bias_text     ↔  opposing attribute pairs ("female"/"male")
  candidate_prompt ↔  debias_vl crossed grid (class × spurious)
  spurious_prompt  ↔  pure spurious descriptions ("water background")
  S_pairs       ↔  same class, different spurious (semantic preservation)
  B_pairs       ↔  same spurious, different class (bias direction)
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
]

# ── debias_vl word-pair grid: sentiment × topic ────────────────────────
# 3 sentiments × 6 topics = 18 crossed prompts
candidate_prompt = [
    # Negative × 6 topics
    "A negative view on cryptocurrency.",                # 0
    "A negative view on corporate earnings.",            # 1
    "A negative view on the stock market.",              # 2
    "A negative view on economic policy.",               # 3
    "A negative view on a specific company.",            # 4
    "A negative view on interest rates.",                # 5
    # Neutral × 6 topics
    "A neutral observation about cryptocurrency.",       # 6
    "A neutral observation about corporate earnings.",   # 7
    "A neutral observation about the stock market.",     # 8
    "A neutral observation about economic policy.",      # 9
    "A neutral observation about a specific company.",   # 10
    "A neutral observation about interest rates.",       # 11
    # Positive × 6 topics
    "A positive view on cryptocurrency.",                # 12
    "A positive view on corporate earnings.",            # 13
    "A positive view on the stock market.",              # 14
    "A positive view on economic policy.",               # 15
    "A positive view on a specific company.",            # 16
    "A positive view on interest rates.",                # 17
]

# Pure topic descriptions (no sentiment)
spurious_prompt = [
    "A statement about cryptocurrency.",
    "A statement about corporate earnings.",
    "A statement about the stock market.",
    "A statement about economic policy.",
    "A statement about a specific company.",
    "A statement about interest rates.",
]

# S pairs: same sentiment, different topic → PRESERVE sentiment
S_pairs = []
for _offset in [0, 6, 12]:
    for _i in range(6):
        for _j in range(_i + 1, 6):
            S_pairs.append([_offset + _i, _offset + _j])
# 15 pairs × 3 sentiments = 45 total

# B pairs: same topic, different sentiment → BIAS to remove
B_pairs = []
for _topic in range(6):
    for _s1, _s2 in [(0, 6), (0, 12), (6, 12)]:
        B_pairs.append([_s1 + _topic, _s2 + _topic])
# 3 pairs × 6 topics = 18 total


def encode_all_prompts(encoder) -> dict:
    """Encode every prompt set with the given encoder.

    Returns dict with keys:
        cls_cb       (3, H)   sentiment class embeddings
        test_cb      (6, H)   neutral concept embeddings
        candidate_cb (18, H)  sentiment × topic crossed
        spurious_cb  (6, H)   pure topic embeddings
    """
    return {
        "cls_cb":       encoder.encode_text(cls_text),
        "test_cb":      encoder.encode_text(test_text),
        "candidate_cb": encoder.encode_text(candidate_prompt),
        "spurious_cb":  encoder.encode_text(spurious_prompt),
    }
