"""
Prompt banks for CBDC sentiment debiasing.

The static prompt roles stay the same:
  cls_text_groups   ↔ class concepts pooled into sentiment prototypes
  target_text       ↔ prompts attacked by PGD and matched to class prototypes
  keep_text         ↔ neutral concept prompts used only for L_s

For the debias_vl stage, topics can now be mined from the training split:
  candidate_prompt  ↔ sentiment × mined-topic crossed grid
  spurious_prompt   ↔ mined-topic prompts without sentiment
  S_pairs           ↔ same sentiment, different topic
  B_pairs           ↔ same topic, different sentiment

If mining fails or the cached split is unavailable, the code falls back to
the default topic list so the pipeline remains runnable.

All prompt templates are parameterized by `text_unit` (e.g., "text", "tweet",
"review") so the pipeline works across domains without code changes.
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
_CLEAN_TOKEN_RE = re.compile(r"^(?:[$#][a-z0-9_]+|[a-z0-9][a-z0-9'&+-]*)$")
_MAX_NGRAM = 3
_ENTITY_TOKEN = "<entity>"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours",
    "rt", "she", "so", "that", "the", "their", "them", "they", "this", "to",
    "too", "us", "was", "we", "were", "will", "with", "you", "your",
}

_GENERIC_BANNED = {
    "about", "all", "back", "best", "better", "can", "day", "days", "don",
    "game", "games", "get", "good", "great", "how", "just", "like", "lot",
    "lots", "many", "month", "months", "more", "much", "need", "night",
    "now", "one", "out", "play", "playing", "really", "see", "some", "still",
    "story", "there", "thing", "things", "time", "today", "tomorrow", "well",
    "what", "work", "year", "years",
}
_PLACEHOLDER_TOKENS = {"<url>", "<other_user>", "<user>", "<person>", "<number>", "[unk]", "unk"}
_CONTRACTION_TAILS = {"'s", "'t", "'m", "'re", "'ve", "'ll", "'d", "s", "t", "m", "re", "ve", "ll", "d"}


# ── Factory functions for text_unit-parameterized prompts ─────────────────

def make_cls_text_groups(text_unit: str = "text") -> list[list[str]]:
    """Sentiment class concept prompts, parameterized by text_unit."""
    return [
        [
            f"A {text_unit} expressing negative sentiment.",
            "This text conveys unfavorable or unhappy sentiment.",
            f"The writer sounds negative about the topic or entity.",
            f"A pessimistic {text_unit}.",
        ],
        [
            f"A {text_unit} expressing neutral sentiment.",
            "This text is informational and emotionally neutral.",
            f"The writer sounds balanced and non-committal about the topic or entity.",
            f"A factual {text_unit} without strong sentiment.",
        ],
        [
            f"A {text_unit} expressing positive sentiment.",
            "This text conveys favorable or enthusiastic sentiment.",
            f"The writer sounds positive about the topic or entity.",
            f"An optimistic {text_unit}.",
        ],
    ]


def make_target_text(text_unit: str = "text") -> list[str]:
    """PGD target prompts for RN50-style text_iccv matching."""
    return [
        f"A negative-sentiment {text_unit}.",
        f"A neutral-sentiment {text_unit}.",
        f"A positive-sentiment {text_unit}.",
    ]


def make_keep_text(text_unit: str = "text") -> list[str]:
    """Neutral concept prompts for L_s preservation.

    These should represent generic tweet content that is sentiment-neutral,
    matching the actual register and topics of the TSAD dataset.
    """
    return [
        f"Someone sharing a personal update.",
        f"A casual remark about daily life.",
        f"A short message to a friend.",
        f"Someone talking about their day.",
        f"A comment about plans for later.",
        f"A brief observation about something.",
        f"Someone mentioning what they are doing.",
        f"A quick update about the weekend.",
        f"A message about something that happened.",
        f"Someone replying to another person.",
        f"A {text_unit} about an everyday topic.",
        f"A conversational {text_unit} online.",
    ]



# ── Data-grounded default topics ───────────────────────────────────────────
#
# Design rationale (following debias_vl / CBDC paper):
#   In Waterbirds, spurious topics are actual confound attributes in the data
#   (land/water background). In CelebA, they are gender. The topics must
#   capture attributes that (a) actually appear in the corpus and (b) are
#   balanced across sentiment classes (high label entropy).
#
#   Analysis of TSAD tweets (27,480 samples) reveals two confound types:
#
#   1. CONTENT TOPICS — what tweets are about. These are personal/daily-life
#      themes (work, school, friends, movies), not news categories. Selected
#      by: frequency > 80, label entropy > 0.93 across neg/neu/pos.
#
#   2. STYLE CONFOUNDS — how tweets are written. These are the strongest
#      spurious correlations in the data:
#        - Positive emoticons (:) :D) → 56% positive (entropy 0.88)
#        - Ending with question mark → 59% neutral (entropy 0.87)
#        - Containing a URL/link → 47% neutral (entropy 0.94)
#        - Very short (≤4 words) → 52% neutral (entropy 0.93)
#        - Internet laughter (lol, haha) → skews positive (entropy 0.93)
#      These are analogous to "background texture" in Waterbirds: features
#      that co-occur with labels but are not causally related to sentiment.
#
#   The prompt template uses "about {topic}" for content topics and
#   "written with/in {style}" for style confounds, following the pattern
#   of debias_vl's per-attribute template specialization.

# Content topics: balanced across sentiment, frequent in TSAD tweets
_CONTENT_TOPICS = [
    # Daily routine (entropy 0.93-0.99, freq 100-1100)
    ("work", "content"),
    ("school", "content"),
    ("sleep", "content"),
    ("home", "content"),
    ("food", "content"),
    ("coffee", "content"),
    # Social / personal (entropy 0.94-1.00, freq 100-360)
    ("friends", "content"),
    ("family", "content"),
    ("people", "content"),
    ("party", "content"),
    # Media / entertainment (entropy 0.93-0.97, freq 100-340)
    ("watching something", "content"),
    ("a movie", "content"),
    ("a show", "content"),
    ("music", "content"),
    # Time / plans (entropy 0.94-0.99, freq 200-510)
    ("plans for tomorrow", "content"),
    ("the weekend", "content"),
    ("last night", "content"),
    ("today", "content"),
    # Communication (entropy 0.94-0.99, freq 150-530)
    ("twitter", "content"),
    ("a phone call", "content"),
]

# Style confounds: writing patterns correlated with sentiment
_STYLE_TOPICS = [
    # Strong confounds (entropy < 0.93)
    ("emoticons and smiley faces", "style"),         # 56% positive
    ("a question", "style"),                          # 59% neutral
    ("a link or URL", "style"),                       # 47% neutral
    ("internet slang like lol or haha", "style"),     # skews pos/neu
    ("very few words", "style"),                      # 52% neutral
    # Moderate confounds (entropy 0.93-0.97)
    ("many exclamation marks", "style"),              # 45% positive
    ("words in all caps", "style"),                   # balanced but distinctive
    ("first person language", "style"),               # 44% negative
    ("trailing off with ellipsis", "style"),          # balanced
    ("elongated words like sooo or yesss", "style"),  # balanced
    ("informal conversational language", "style"),    # general register
    ("direct and formal language", "style"),          # contrast register
]

# Combined default list — content first, then style confounds
DEFAULT_TOPICS = [t[0] for t in _CONTENT_TOPICS + _STYLE_TOPICS]

# Metadata for template selection (content vs style)
DEFAULT_TOPIC_METADATA = [
    {"topic": topic, "kind": kind} for topic, kind in _CONTENT_TOPICS + _STYLE_TOPICS
]

# ── Financial domain topics (preserved for --text_unit tweet / finance) ───
FINANCE_TOPICS = [
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


def _normalize_optional_text(value):
    if value is None:
        return None
    try:
        if isinstance(value, float) and value != value:
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text if text else None


def _normalize_entity_text(entity: str | None) -> str | None:
    entity = _normalize_optional_text(entity)
    if entity is None:
        return None
    entity = entity.replace("_", " ").strip()
    entity = re.sub(r"\s+", " ", entity)
    return entity if entity else None


def _normalize_cleaned_tokens(tokens: Sequence[str] | None) -> list[str]:
    if not tokens:
        return []
    normalized = []
    for tok in tokens:
        tok = str(tok).strip().lower().replace("'", "'")
        if not tok or tok in {",", ".", "!", "?", ";", ":", "''", "``", "'", "\"", "[", "]", "(", ")"}:
            continue
        if tok in _PLACEHOLDER_TOKENS:
            continue
        if tok == "<entity>":
            tok = _ENTITY_TOKEN
        elif not _CLEAN_TOKEN_RE.match(tok):
            continue
        if tok in _CONTRACTION_TAILS:
            continue
        normalized.append(tok)
    return normalized


def _tokens_from_record(record: dict) -> list[str]:
    cleaned_tokens = _normalize_cleaned_tokens(record.get("cleaned_tokens"))
    if cleaned_tokens:
        tokens = [tok for tok in cleaned_tokens if tok != _ENTITY_TOKEN]
    else:
        tokens = _tokenize_tweet_text(record["text"])

    selected_text = record.get("selected_text")
    if selected_text:
        selected_tokens = set(_tokenize_tweet_text(selected_text))
        if selected_tokens:
            tokens = [tok for tok in tokens if tok not in selected_tokens]

    return [tok for tok in tokens if tok not in _STOPWORDS]


def _is_valid_phrase(tokens: Sequence[str]) -> bool:
    if not tokens or len(tokens) > _MAX_NGRAM:
        return False
    if any(tok.startswith("<") and tok.endswith(">") for tok in tokens):
        return False
    if all(tok in _STOPWORDS for tok in tokens):
        return False
    if any(tok in _GENERIC_BANNED for tok in tokens):
        return False
    if any(tok in _CONTRACTION_TAILS for tok in tokens):
        return False
    if len(tokens) > 1 and (tokens[0] in _STOPWORDS or tokens[-1] in _STOPWORDS):
        return False
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in _STOPWORDS:
            return False
        if tok in _GENERIC_BANNED:
            return False
        if not tok.startswith(("$", "#")):
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


def _score_topic_row(total, counts, *, specificity_bonus=1.0) -> float:
    entropy = _label_entropy(counts)
    return math.log1p(total) * entropy * specificity_bonus


def mine_entity_topics_from_records(
    records: Sequence[dict],
    max_topics: int = 12,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
) -> list[dict]:
    """Mine balanced high-frequency entities as debias_vl topics."""
    n_docs = len(records)
    max_doc_freq = max(min_doc_freq + 1, int(max_doc_freq_ratio * n_docs))

    doc_counts = Counter()
    label_doc_counts = defaultdict(lambda: [0, 0, 0])
    display_forms = defaultdict(Counter)

    for record in records:
        entity = _normalize_entity_text(record.get("entity"))
        if entity is None:
            continue
        norm = entity.lower()
        doc_counts[norm] += 1
        label_doc_counts[norm][int(record["label"])] += 1
        display_forms[norm][entity] += 1

    rows = []
    for norm_entity, total in doc_counts.items():
        if total < min_doc_freq or total > max_doc_freq:
            continue
        counts = label_doc_counts[norm_entity]
        entropy = _label_entropy(counts)
        if entropy < 0.35:
            continue
        display = display_forms[norm_entity].most_common(1)[0][0]
        rows.append(
            {
                "topic": display,
                "kind": "entity",
                "source": "entity",
                "doc_freq": int(total),
                "label_counts": [int(x) for x in counts],
                "entropy": float(entropy),
                "score": float(_score_topic_row(total, counts, specificity_bonus=1.15)),
            }
        )

    rows.sort(key=lambda row: (row["score"], row["doc_freq"]), reverse=True)
    return rows[:max_topics]


def mine_phrase_topics_from_records(
    records: Sequence[dict],
    max_topics: int = 32,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
) -> list[dict]:
    """Mine label-balanced topic/style phrases from cleaned/context tokens.

    Phrases are ranked by:
      frequency × class-balance entropy × small specificity bonus
    and then greedily deduplicated for diversity.
    """
    doc_counts = Counter()
    label_doc_counts = defaultdict(lambda: [0, 0, 0])
    n_docs = len(records)
    max_doc_freq = max(min_doc_freq + 1, int(max_doc_freq_ratio * n_docs))

    for record in records:
        label = int(record["label"])
        phrases = _iter_phrase_candidates(_tokens_from_record(record))
        for phrase in phrases:
            doc_counts[phrase] += 1
            label_doc_counts[phrase][label] += 1

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
        specificity_bonus = 1.0 + 0.25 * (ngram_len - 1) + 0.20 * has_symbol
        score = _score_topic_row(total, counts, specificity_bonus=specificity_bonus)

        scored.append(
            {
                "topic": phrase,
                "kind": "phrase",
                "source": "cleaned_context",
                "doc_freq": int(total),
                "label_counts": [int(x) for x in counts],
                "entropy": float(entropy),
                "score": float(score),
            }
        )

    scored.sort(key=lambda row: (row["score"], row["doc_freq"]), reverse=True)

    chosen = []
    chosen_topics = []
    for row in scored:
        if _is_redundant_phrase(row["topic"], chosen_topics):
            continue
        chosen.append(row)
        chosen_topics.append(row["topic"])
        if len(chosen) >= max_topics:
            break

    return chosen


def _records_from_cached_train(tokenizer, train_data: dict) -> list[dict]:
    texts = train_data.get("texts")
    if texts is None:
        if tokenizer is None:
            raise ValueError("Tokenizer is required to decode cached train split without raw texts.")
        texts = tokenizer.batch_decode(train_data["input_ids"], skip_special_tokens=True)

    labels = train_data["labels"].tolist()
    entities = train_data.get("entities") or [None] * len(texts)
    cleaned_tokens = train_data.get("cleaned_tokens") or [None] * len(texts)
    selected_texts = train_data.get("selected_texts") or [None] * len(texts)
    time_of_tweet = train_data.get("time_of_tweet") or [None] * len(texts)
    age_of_user = train_data.get("age_of_user") or [None] * len(texts)
    country = train_data.get("country") or [None] * len(texts)

    return [
        {
            "text": texts[i],
            "label": int(labels[i]),
            "entity": entities[i],
            "cleaned_tokens": cleaned_tokens[i],
            "selected_text": selected_texts[i],
            "time_of_tweet": time_of_tweet[i],
            "age_of_user": age_of_user[i],
            "country": country[i],
        }
        for i in range(len(texts))
    ]


def _has_structured_mining_fields(records: Sequence[dict]) -> bool:
    if not records:
        return False
    has_entity = any(record.get("entity") for record in records)
    has_cleaned = any(record.get("cleaned_tokens") for record in records)
    has_selected = any(record.get("selected_text") for record in records)
    return has_entity or has_cleaned or has_selected


def _load_records_for_mining(tokenizer, cache_dir: str) -> tuple[list[dict] | None, str]:
    train_path = os.path.join(cache_dir, "z_tweet_train.pt")
    if os.path.exists(train_path):
        train_data = torch.load(train_path, map_location="cpu")
        records = _records_from_cached_train(tokenizer, train_data)
        if _has_structured_mining_fields(records):
            return records, f"cache:{train_path}"

    try:
        from dataset import load_records

        train_records, _, _ = load_records()
        if train_records:
            return train_records, "dataset:load_records"
    except Exception:
        pass

    if os.path.exists(train_path):
        train_data = torch.load(train_path, map_location="cpu")
        return _records_from_cached_train(tokenizer, train_data), f"cache-text-only:{train_path}"

    return None, "none"


def combine_topic_rows(
    entity_rows: Sequence[dict],
    phrase_rows: Sequence[dict],
    max_topics: int,
) -> list[dict]:
    """Prefer structured entity topics, then fill with diverse phrase topics."""
    combined = []
    chosen_topics = []

    entity_budget = min(max_topics // 2, 12)
    for row in entity_rows[:entity_budget]:
        if _is_redundant_phrase(row["topic"], chosen_topics):
            continue
        combined.append(row)
        chosen_topics.append(row["topic"])

    for row in phrase_rows:
        if _is_redundant_phrase(row["topic"], chosen_topics):
            continue
        combined.append(row)
        chosen_topics.append(row["topic"])
        if len(combined) >= max_topics:
            break

    if len(combined) < max_topics:
        for row in entity_rows[entity_budget:]:
            if _is_redundant_phrase(row["topic"], chosen_topics):
                continue
            combined.append(row)
            chosen_topics.append(row["topic"])
            if len(combined) >= max_topics:
                break

    combined.sort(key=lambda row: row["score"], reverse=True)
    return combined[:max_topics]


def mine_topic_phrases_from_cache(
    tokenizer,
    cache_dir: str | None = None,
    max_topics: int = 32,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
    force_refresh: bool = False,
) -> tuple[list[str], list[dict], bool]:
    """Mine topic phrases from cached train data and persist the result."""
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

    records, record_source = _load_records_for_mining(tokenizer, cache_dir)
    if not records:
        topics = DEFAULT_TOPICS[:max_topics]
        metadata = []
        with open(out_path, "w") as f:
            json.dump(
                {
                    "using_mined_topics": False,
                    "topics": topics,
                    "metadata": metadata,
                    "reason": f"no mining records available ({record_source})",
                },
                f,
                indent=2,
            )
        return topics, metadata, False

    entity_rows = mine_entity_topics_from_records(
        records,
        max_topics=max_topics,
        min_doc_freq=max(5, min_doc_freq // 2),
        max_doc_freq_ratio=max_doc_freq_ratio,
    )
    phrase_rows = mine_phrase_topics_from_records(
        records,
        max_topics=max_topics,
        min_doc_freq=min_doc_freq,
        max_doc_freq_ratio=max_doc_freq_ratio,
    )
    metadata = combine_topic_rows(entity_rows, phrase_rows, max_topics=max_topics)
    topics = [row["topic"] for row in metadata]

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
                "record_source": record_source,
            },
            f,
            indent=2,
        )

    return topics, metadata, using_mined_topics


def build_prompt_bank(
    topics: Sequence[str],
    topic_metadata: Sequence[dict] | None = None,
    using_mined_topics: bool = False,
    text_unit: str = "text",
) -> dict:
    """Build the debias_vl prompt grid and index pairs for a topic list."""
    topics = list(topics)
    n_topics = len(topics)
    meta_by_topic = {}
    for row in topic_metadata or []:
        topic = row.get("topic")
        if topic is not None and topic not in meta_by_topic:
            meta_by_topic[topic] = row

    # Build text_unit-parameterized prompt sets
    _cls_text_groups = make_cls_text_groups(text_unit)
    _cls_group_sizes = [len(group) for group in _cls_text_groups]
    _cls_text = [group[0] for group in _cls_text_groups]
    _target_text = make_target_text(text_unit)
    _keep_text = make_keep_text(text_unit)

    def _topic_templates(topic: str):
        meta = meta_by_topic.get(topic, {})
        kind = meta.get("kind", "content")
        if kind == "entity":
            return {
                "spurious": f"A {text_unit} mentioning {topic}.",
                "negative": f"A negative {text_unit} mentioning {topic}.",
                "neutral": f"A neutral {text_unit} mentioning {topic}.",
                "positive": f"A positive {text_unit} mentioning {topic}.",
            }
        if kind == "style":
            # Style confounds use "written with/in" to describe HOW the text
            # is written, analogous to "with {background}" in Waterbirds.
            return {
                "spurious": f"A {text_unit} written with {topic}.",
                "negative": f"A negative {text_unit} written with {topic}.",
                "neutral": f"A neutral {text_unit} written with {topic}.",
                "positive": f"A positive {text_unit} written with {topic}.",
            }
        if kind == "content":
            # Content topics use "about" to describe WHAT the text is about.
            return {
                "spurious": f"A {text_unit} about {topic}.",
                "negative": f"A negative {text_unit} about {topic}.",
                "neutral": f"A neutral {text_unit} about {topic}.",
                "positive": f"A positive {text_unit} about {topic}.",
            }
        # Fallback for mined phrase topics
        return {
            "spurious": f"A {text_unit} using the context phrase {topic}.",
            "negative": f"A negative {text_unit} using the context phrase {topic}.",
            "neutral": f"A neutral {text_unit} using the context phrase {topic}.",
            "positive": f"A positive {text_unit} using the context phrase {topic}.",
        }

    templates = [_topic_templates(topic) for topic in topics]
    candidate_prompt = (
        [tpl["negative"] for tpl in templates]
        + [tpl["neutral"] for tpl in templates]
        + [tpl["positive"] for tpl in templates]
    )
    spurious_prompt = [tpl["spurious"] for tpl in templates]

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
        "text_unit": text_unit,
        "cls_text_groups": _cls_text_groups,
        "cls_group_sizes": _cls_group_sizes,
        "cls_text": _cls_text,
        "target_text": _target_text,
        "keep_text": _keep_text,
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
    text_unit: str = "text",
) -> dict:
    """Return the active prompt bank, preferring mined phrases."""
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
                text_unit=text_unit,
            )
        except Exception as exc:
            bank = build_prompt_bank(
                DEFAULT_TOPICS[:max_topics],
                topic_metadata=DEFAULT_TOPIC_METADATA[:max_topics],
                using_mined_topics=False,
                text_unit=text_unit,
            )
            bank["mining_error"] = str(exc)
            return bank

    return build_prompt_bank(
        DEFAULT_TOPICS[:max_topics],
        topic_metadata=DEFAULT_TOPIC_METADATA[:max_topics],
        using_mined_topics=False,
        text_unit=text_unit,
    )


DEFAULT_PROMPT_BANK = build_prompt_bank(
    DEFAULT_TOPICS,
    topic_metadata=DEFAULT_TOPIC_METADATA,
    using_mined_topics=False,
    text_unit="text",
)


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
