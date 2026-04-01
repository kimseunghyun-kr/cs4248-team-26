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
TOPIC_MINER_VERSION = 2

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
_EMOTICON_RE = re.compile(r"(?i)(?:[:;=8x][-^']?[)(dpo/\\\\]|<3)")
_LAUGHTER_RE = re.compile(r"(?i)\b(?:lol|lmao|lmfao|rofl|haha|hehe)\b")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")
_ELONGATED_RE = re.compile(r"(?i)\b\w*([a-z])\1{2,}\w*\b")


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

LEGACY_STYLE_BIASES = [
    "ending with a question mark",
    "containing multiple question marks",
    "containing an exclamation mark",
    "that is very short",
    "containing internet laughter like lol",
    "containing an emoticon",
]

_DEFAULT_CONTENT_PATTERNS = [
    {"topic": "work", "kind": "content", "aliases": ["work", "working", "job", "office", "shift"]},
    {"topic": "school", "kind": "content", "aliases": ["school", "class", "classes", "college", "exam", "study", "homework"]},
    {"topic": "sleep", "kind": "content", "aliases": ["sleep", "slept", "bed", "nap", "naps"]},
    {"topic": "home", "kind": "content", "aliases": ["home", "house"]},
    {"topic": "food", "kind": "content", "aliases": ["food", "eat", "eating", "dinner", "lunch", "breakfast", "meal"]},
    {"topic": "coffee", "kind": "content", "aliases": ["coffee", "latte", "espresso", "starbucks", "caffeine"]},
    {"topic": "friends", "kind": "content", "aliases": ["friend", "friends", "buddy", "buddies"]},
    {"topic": "family", "kind": "content", "aliases": ["family", "mom", "mother", "dad", "father", "parents", "sister", "brother"]},
    {"topic": "people", "kind": "content", "aliases": ["people", "person", "everyone", "somebody", "anyone"]},
    {"topic": "party", "kind": "content", "aliases": ["party", "partying", "club", "clubbing"]},
    {"topic": "watching something", "kind": "content", "aliases": ["watching"]},
    {"topic": "a movie", "kind": "content", "aliases": ["movie", "movies", "film", "films", "cinema"]},
    {"topic": "a show", "kind": "content", "aliases": ["show", "shows", "episode", "episodes", "season", "series", "tv"]},
    {"topic": "music", "kind": "content", "aliases": ["music", "song", "songs", "album", "albums", "band", "bands", "concert"]},
    {"topic": "plans for tomorrow", "kind": "content", "aliases": ["tomorrow"]},
    {"topic": "the weekend", "kind": "content", "aliases": ["weekend", "saturday", "sunday"]},
    {"topic": "last night", "kind": "content", "aliases": ["last night"]},
    {"topic": "today", "kind": "content", "aliases": ["today", "tonight"]},
    {"topic": "twitter", "kind": "content", "aliases": ["twitter", "tweet", "tweets", "tweeting"]},
    {"topic": "a phone call", "kind": "content", "aliases": ["phone", "call", "called", "voicemail"]},
]

_DEFAULT_STYLE_PATTERNS = [
    {"topic": "emoticons and smiley faces", "kind": "style"},
    {"topic": "a question", "kind": "style"},
    {"topic": "a link or URL", "kind": "style"},
    {"topic": "internet slang like lol or haha", "kind": "style"},
    {"topic": "very few words", "kind": "style"},
    {"topic": "many exclamation marks", "kind": "style"},
    {"topic": "words in all caps", "kind": "style"},
    {"topic": "first person language", "kind": "style"},
    {"topic": "trailing off with ellipsis", "kind": "style"},
    {"topic": "elongated words like sooo or yesss", "kind": "style"},
    {"topic": "informal conversational language", "kind": "style"},
    {"topic": "direct and formal language", "kind": "style"},
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


def _raw_record_tokens(record: dict) -> list[str]:
    return _tokenize_tweet_text(record["text"])


def _topic_alias_match(tokens: set[str], alias: str) -> bool:
    alias_tokens = [tok for tok in _tokenize_tweet_text(alias) if tok not in _STOPWORDS]
    if not alias_tokens:
        return False
    return all(tok in tokens for tok in alias_tokens)


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


def mine_curated_topic_rows_from_records(
    records: Sequence[dict],
    max_topics: int = 32,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
) -> list[dict]:
    """Score the curated TSAD topic/style bank against the actual dataset."""
    n_docs = len(records)
    max_doc_freq = max(min_doc_freq + 1, int(max_doc_freq_ratio * n_docs))
    rows = []

    for spec in _DEFAULT_CONTENT_PATTERNS:
        counts = [0, 0, 0]
        total = 0
        for record in records:
            token_set = set(_tokens_from_record(record))
            if any(_topic_alias_match(token_set, alias) for alias in spec["aliases"]):
                counts[int(record["label"])] += 1
                total += 1
        if total < min_doc_freq or total > max_doc_freq:
            continue
        entropy = _label_entropy(counts)
        if entropy < 0.30:
            continue
        rows.append(
            {
                "topic": spec["topic"],
                "kind": spec["kind"],
                "source": "dataset_pattern",
                "doc_freq": int(total),
                "label_counts": [int(x) for x in counts],
                "entropy": float(entropy),
                "score": float(_score_topic_row(total, counts, specificity_bonus=1.05)),
            }
        )

    informal_markers = {"gonna", "wanna", "gotta", "omg", "ugh", "ya", "yall", "dunno"}
    first_person_markers = {"i", "im", "i'm", "ive", "i've", "me", "my", "mine"}
    formal_markers = {"please", "regarding", "appreciate", "thank", "thanks", "sincerely"}

    for spec in _DEFAULT_STYLE_PATTERNS:
        counts = [0, 0, 0]
        total = 0
        for record in records:
            raw_text = str(record["text"])
            raw_tokens = set(_raw_record_tokens(record))

            if spec["topic"] == "emoticons and smiley faces":
                matched = bool(_EMOTICON_RE.search(raw_text))
            elif spec["topic"] == "a question":
                matched = "?" in raw_text
            elif spec["topic"] == "a link or URL":
                matched = bool(_URL_RE.search(raw_text))
            elif spec["topic"] == "internet slang like lol or haha":
                matched = bool(_LAUGHTER_RE.search(raw_text))
            elif spec["topic"] == "very few words":
                matched = len(_raw_record_tokens(record)) <= 4
            elif spec["topic"] == "many exclamation marks":
                matched = raw_text.count("!") >= 2
            elif spec["topic"] == "words in all caps":
                matched = bool(_ALL_CAPS_RE.search(raw_text))
            elif spec["topic"] == "first person language":
                matched = bool(first_person_markers & raw_tokens)
            elif spec["topic"] == "trailing off with ellipsis":
                matched = bool(_ELLIPSIS_RE.search(raw_text))
            elif spec["topic"] == "elongated words like sooo or yesss":
                matched = bool(_ELONGATED_RE.search(raw_text))
            elif spec["topic"] == "informal conversational language":
                matched = bool(informal_markers & raw_tokens)
            elif spec["topic"] == "direct and formal language":
                matched = bool(formal_markers & raw_tokens)
            else:
                matched = False

            if matched:
                counts[int(record["label"])] += 1
                total += 1

        if total < min_doc_freq or total > max_doc_freq:
            continue
        entropy = _label_entropy(counts)
        if entropy < 0.25:
            continue
        rows.append(
            {
                "topic": spec["topic"],
                "kind": spec["kind"],
                "source": "dataset_pattern",
                "doc_freq": int(total),
                "label_counts": [int(x) for x in counts],
                "entropy": float(entropy),
                "score": float(_score_topic_row(total, counts, specificity_bonus=1.0)),
            }
        )

    rows.sort(key=lambda row: (row["score"], row["doc_freq"]), reverse=True)
    return rows[:max_topics]


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
    curated_rows: Sequence[dict],
    max_topics: int,
) -> list[dict]:
    """Prefer structured entities, then mined phrases, then curated dataset-matched rows."""
    combined = []
    chosen_topics = []

    entity_budget = min(max_topics // 2, 12)
    for row in entity_rows[:entity_budget]:
        if _is_redundant_phrase(row["topic"], chosen_topics):
            continue
        combined.append(row)
        chosen_topics.append(row["topic"])

    curated_content_rows = [row for row in curated_rows if row.get("kind") != "style"]
    curated_style_rows = [row for row in curated_rows if row.get("kind") == "style"]

    if len(combined) < max_topics:
        for row in curated_content_rows:
            if _is_redundant_phrase(row["topic"], chosen_topics):
                continue
            combined.append(row)
            chosen_topics.append(row["topic"])
            if len(combined) >= max_topics:
                break

    if len(combined) < max_topics:
        for row in phrase_rows:
            if _is_redundant_phrase(row["topic"], chosen_topics):
                continue
            combined.append(row)
            chosen_topics.append(row["topic"])
            if len(combined) >= max_topics:
                break

    if len(combined) < max_topics:
        for row in curated_style_rows:
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
        cached_version = payload.get("miner_version", 0)
        topics = payload.get("topics", [])
        metadata = payload.get("metadata", [])
        using_mined = bool(payload.get("using_mined_topics", False))
        if cached_version == TOPIC_MINER_VERSION and topics and (using_mined or metadata):
            return topics[:max_topics], metadata[:max_topics], using_mined

    records, record_source = _load_records_for_mining(tokenizer, cache_dir)
    if not records:
        topics = DEFAULT_TOPICS[:max_topics]
        metadata = []
        with open(out_path, "w") as f:
            json.dump(
                {
                    "miner_version": TOPIC_MINER_VERSION,
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
    curated_rows = mine_curated_topic_rows_from_records(
        records,
        max_topics=max_topics,
        min_doc_freq=min_doc_freq,
        max_doc_freq_ratio=max_doc_freq_ratio,
    )
    metadata = combine_topic_rows(entity_rows, phrase_rows, curated_rows, max_topics=max_topics)
    topics = [row["topic"] for row in metadata]

    using_mined_topics = len(topics) >= min(8, max_topics)
    if not using_mined_topics:
        topics = DEFAULT_TOPICS[:max_topics]
        metadata = []

    with open(out_path, "w") as f:
        json.dump(
            {
                "miner_version": TOPIC_MINER_VERSION,
                "using_mined_topics": using_mined_topics,
                "topics": topics,
                "metadata": metadata,
                "record_source": record_source,
                "phrase_topic_count": len(phrase_rows),
                "entity_topic_count": len(entity_rows),
                "curated_topic_count": len(curated_rows),
                "reason": None if using_mined_topics else "insufficient dataset-supported mined topics",
            },
            f,
            indent=2,
        )

    return topics, metadata, using_mined_topics
# ---------------------------------------------------------------------------
# Core Prompt Builders
# ---------------------------------------------------------------------------

CBDC_STYLE_BIAS_PAIRS = [
    (
        "A {text_unit} ending with a question mark.",
        "A {text_unit} written as a statement.",
        "question-mark vs statement",
    ),
    (
        "A {text_unit} containing multiple question marks.",
        "A {text_unit} with standard punctuation.",
        "repeated-question-marks vs standard punctuation",
    ),
    (
        "A {text_unit} containing an exclamation mark.",
        "A {text_unit} without exclamation marks.",
        "exclamation-mark vs no-exclamation-mark",
    ),
    (
        "A {text_unit} that is very short.",
        "A longer and more detailed {text_unit}.",
        "very-short vs longer-detailed",
    ),
    (
        "A {text_unit} containing internet laughter like lol.",
        "A {text_unit} without internet laughter.",
        "internet-laughter vs no-laughter",
    ),
    (
        "A {text_unit} containing an emoticon or smiley face.",
        "A {text_unit} without any emoticon.",
        "emoticon vs no-emoticon",
    ),
]


def _build_debias_candidate_and_pairs(
    topics: Sequence[str],
    topic_metadata: Sequence[dict] | None,
    text_unit: str,
) -> tuple[list[str], list[str], list[tuple[int, int]], list[tuple[int, int]]]:
    sentiments = ["negative", "neutral", "positive"]
    candidate_prompt = []
    spurious_prompt = []
    grid_indices = {}

    def get_prompt(sentiment: str, topic: str, kind: str) -> str:
        if kind == "style":
            return f"A {sentiment} {text_unit} {topic}"
        return f"A {sentiment} {text_unit} about {topic}"

    for idx, topic in enumerate(topics):
        kind = topic_metadata[idx].get("kind", "content") if topic_metadata and idx < len(topic_metadata) else "content"
        if kind == "style":
            spurious_prompt.append(f"A {text_unit} {topic}")
        else:
            spurious_prompt.append(f"A {text_unit} about {topic}")

    flat_idx = 0
    for s_idx, sentiment in enumerate(sentiments):
        for t_idx, topic in enumerate(topics):
            kind = topic_metadata[t_idx].get("kind", "content") if topic_metadata and t_idx < len(topic_metadata) else "content"
            candidate_prompt.append(get_prompt(sentiment, topic, kind))
            grid_indices[(s_idx, t_idx)] = flat_idx
            flat_idx += 1

    same_sentiment_pairs = []
    same_topic_pairs = []
    for s_idx in range(len(sentiments)):
        for t1_idx in range(len(topics)):
            for t2_idx in range(t1_idx + 1, len(topics)):
                same_sentiment_pairs.append((grid_indices[(s_idx, t1_idx)], grid_indices[(s_idx, t2_idx)]))

    for t_idx in range(len(topics)):
        for s1_idx in range(len(sentiments)):
            for s2_idx in range(s1_idx + 1, len(sentiments)):
                same_topic_pairs.append((grid_indices[(s1_idx, t_idx)], grid_indices[(s2_idx, t_idx)]))

    return candidate_prompt, spurious_prompt, same_sentiment_pairs, same_topic_pairs


def build_debias_vl_prompt_bank(
    topics: Sequence[str],
    topic_metadata: Sequence[dict] | None = None,
    using_mined_topics: bool = False,
    text_unit: str = "text",
    mining_error: str | None = None,
) -> dict:
    """Build the prompt bank used by the debias_vl discovery stage."""
    candidate_prompt, spurious_prompt, s_pairs, b_pairs = _build_debias_candidate_and_pairs(
        topics,
        topic_metadata,
        text_unit,
    )
    return {
        "candidate_prompt": candidate_prompt,
        "spurious_prompt": spurious_prompt,
        "S_pairs": s_pairs,
        "B_pairs": b_pairs,
        "topics": list(topics),
        "topic_metadata": list(topic_metadata or []),
        "using_mined_topics": using_mined_topics,
        "mining_error": mining_error,
    }


def build_prompt_bank(
    topics: Sequence[str],
    topic_metadata: Sequence[dict] | None = None,
    using_mined_topics: bool = False,
    text_unit: str = "text",
) -> dict:
    """Backward-compatible prompt-bank builder from the older single-bank file."""
    merged = build_cbdc_prompt_bank(text_unit=text_unit)
    merged.update(
        build_debias_vl_prompt_bank(
            topics=topics,
            topic_metadata=topic_metadata,
            using_mined_topics=using_mined_topics,
            text_unit=text_unit,
        )
    )
    return merged


def build_cbdc_prompt_bank(text_unit: str = "text") -> dict:
    """Build the prompt bank used by the pure CBDC text_iccv stage."""
    pole_a_text = [a.format(text_unit=text_unit) for a, _, _ in CBDC_STYLE_BIAS_PAIRS]
    pole_b_text = [b.format(text_unit=text_unit) for _, b, _ in CBDC_STYLE_BIAS_PAIRS]
    pair_names = [name for _, _, name in CBDC_STYLE_BIAS_PAIRS]
    cls_groups = make_cls_text_groups(text_unit)
    return {
        "cls_text_groups": cls_groups,
        "cls_group_sizes": [len(group) for group in cls_groups],
        "target_text": make_target_text(text_unit),
        "keep_text": make_keep_text(text_unit),
        "bias_pole_a_text": pole_a_text,
        "bias_pole_b_text": pole_b_text,
        "bias_pair_names": pair_names,
        "legacy_style_biases": list(LEGACY_STYLE_BIASES),
    }


def get_debias_vl_prompt_bank(
    tokenizer=None,
    cache_dir: str | None = None,
    use_mined_topics: bool = True,
    max_topics: int = 32,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
    force_refresh: bool = False,
    text_unit: str = "text",
) -> dict:
    """Return the active debias_vl topic bank."""
    metadata: list[dict] = []
    mining_error = None
    using_mined_topics = False
    topics = DEFAULT_TOPICS[:max_topics]

    if use_mined_topics:
        try:
            topics, metadata, using_mined_topics = mine_topic_phrases_from_cache(
                tokenizer,
                cache_dir=cache_dir,
                max_topics=max_topics,
                min_doc_freq=min_doc_freq,
                max_doc_freq_ratio=max_doc_freq_ratio,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            topics = DEFAULT_TOPICS[:max_topics]
            metadata = DEFAULT_TOPIC_METADATA[:max_topics]
            mining_error = str(exc)
        else:
            if not metadata and not using_mined_topics:
                metadata = DEFAULT_TOPIC_METADATA[:len(topics)]
            if not using_mined_topics and mining_error is None:
                mining_error = "insufficient dataset-supported mined topics"
    else:
        topics = DEFAULT_TOPICS[:max_topics]
        metadata = DEFAULT_TOPIC_METADATA[:max_topics]

    return build_debias_vl_prompt_bank(
        topics=topics,
        topic_metadata=metadata,
        using_mined_topics=using_mined_topics,
        text_unit=text_unit,
        mining_error=mining_error,
    )


def get_cbdc_prompt_bank(text_unit: str = "text") -> dict:
    """Return the fixed prompt-only CBDC bank."""
    return build_cbdc_prompt_bank(text_unit=text_unit)


def get_combined_prompt_bank(
    tokenizer=None,
    cache_dir: str | None = None,
    use_mined_topics: bool = True,
    max_topics: int = 32,
    min_doc_freq: int = 20,
    max_doc_freq_ratio: float = 0.20,
    force_refresh: bool = False,
    text_unit: str = "text",
) -> dict:
    """Return the combined bank used by the debias_vl->CBDC method."""
    debias_bank = get_debias_vl_prompt_bank(
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        use_mined_topics=use_mined_topics,
        max_topics=max_topics,
        min_doc_freq=min_doc_freq,
        max_doc_freq_ratio=max_doc_freq_ratio,
        force_refresh=force_refresh,
        text_unit=text_unit,
    )
    cbdc_bank = get_cbdc_prompt_bank(text_unit=text_unit)
    merged = dict(cbdc_bank)
    merged.update(debias_bank)
    return merged


def get_prompt_bank(*args, **kwargs) -> dict:
    """Backward-compatible alias for the combined D3 prompt bank."""
    return get_combined_prompt_bank(*args, **kwargs)


DEFAULT_PROMPT_BANK = get_combined_prompt_bank(text_unit="tweet")


def encode_all_prompts(encoder, prompt_bank: dict | None = None) -> dict:
    """Encode prompt-bank fields that are present in the provided bank."""
    bank = prompt_bank or DEFAULT_PROMPT_BANK
    encoded = {}
    if "cls_text_groups" in bank:
        encoded["cls_cb"] = encode_grouped_prompts(encoder, bank["cls_text_groups"])
    if "target_text" in bank:
        encoded["target_cb"] = encoder.encode_text(bank["target_text"])
    if "keep_text" in bank:
        encoded["keep_cb"] = encoder.encode_text(bank["keep_text"])
    if "candidate_prompt" in bank:
        encoded["candidate_cb"] = encoder.encode_text(bank["candidate_prompt"])
    if "spurious_prompt" in bank:
        encoded["spurious_cb"] = encoder.encode_text(bank["spurious_prompt"])
    if "bias_pole_a_text" in bank:
        encoded["bias_pole_a_cb"] = encoder.encode_text(bank["bias_pole_a_text"])
    if "bias_pole_b_text" in bank:
        encoded["bias_pole_b_cb"] = encoder.encode_text(bank["bias_pole_b_text"])
    return encoded

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
