"""
Shared feature extraction utilities used across all dataset classes.

Imports:
    pip install vaderSentiment afinn nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
"""

import nltk
import numpy as np
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
_afinn = Afinn()

LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}

# Broad POS categories (simplified from Penn Treebank tags)
# NN=noun, VB=verb, JJ=adjective, RB=adverb, PR=pronoun, DT=determiner
POS_CATEGORIES = ['NN', 'VB', 'JJ', 'RB', 'PR', 'DT', 'OTHER']

# Full Penn Treebank POS tagset (as returned by nltk.pos_tag)
PENN_TREEBANK_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
    'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
    'WDT', 'WP', 'WP$', 'WRB', '$', "''", '``', ',', '.', ':', '(', ')', '#',
]


def compute_vader_features(text: str) -> list:
    """
    Returns 4 tweet-level VADER sentiment scores.

    Features (in order):
        [compound, pos, neg, neu]

    Note: VADER is designed for social media text and operates at the
    sentence level. It does not produce reliable per-word scores.
    """
    scores = _vader.polarity_scores(text)
    return [scores['compound'], scores['pos'], scores['neg'], scores['neu']]


def compute_afinn_features(text: str) -> list:
    """
    Returns 5 tweet-level AFINN sentiment features.

    Features (in order):
        [total_score, positive_word_count, negative_word_count,
         positive_total_score, negative_total_score]

    - total_score           : sum of all word scores (net sentiment, can be negative)
    - positive_word_count   : number of words with score > 0
    - negative_word_count   : number of words with score < 0
    - positive_total_score  : sum of scores for positive words only (intensity of positivity)
    - negative_total_score  : sum of scores for negative words only (intensity of negativity, <= 0)

    positive_total_score / negative_total_score mirror VADER's scores['pos'] / scores['neg'],
    capturing not just how many sentiment words exist but how strong they are.

    AFINN assigns a score to each known word (-5 to +5).
    Unknown words are scored 0.
    """
    tokens = text.split()
    word_scores = [_afinn.score(w) for w in tokens]
    total = sum(word_scores)
    pos_count = sum(1 for s in word_scores if s > 0)
    neg_count = sum(1 for s in word_scores if s < 0)
    pos_total = sum(s for s in word_scores if s > 0)
    neg_total = sum(s for s in word_scores if s < 0)
    return [total, float(pos_count), float(neg_count), pos_total, neg_total]


def compute_pos_broad_category_counts(text: str) -> list:
    """
    Returns counts of broad POS categories per tweet.
    Used by classical models (NB, LR) as tweet-level features.

    Features (in order):
        [NN_count, VB_count, JJ_count, RB_count, PR_count, DT_count, OTHER_count]

    WARNING for NB users: these are non-negative counts, compatible with
    MultinomialNB. But if you combine them with VADER/AFINN (which can be
    negative), use GaussianNB or ComplementNB instead.
    """
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    counts = {c: 0 for c in POS_CATEGORIES}
    for _, tag in tags:
        if tag.startswith('NN'):
            counts['NN'] += 1
        elif tag.startswith('VB'):
            counts['VB'] += 1
        elif tag.startswith('JJ'):
            counts['JJ'] += 1
        elif tag.startswith('RB'):
            counts['RB'] += 1
        elif tag.startswith('PR'):
            counts['PR'] += 1
        elif tag.startswith('DT'):
            counts['DT'] += 1
        else:
            counts['OTHER'] += 1
    return [float(counts[c]) for c in POS_CATEGORIES]


def compute_pos_specific_tag_counts(text: str) -> list:
    """
    Returns counts of every Penn Treebank POS tag per tweet.
    Used by classical models (NB, LR) when fine-grained POS distinctions matter
    (e.g. separating modals MD, interjections UH, or verb subtypes VBD/VBG/VBN).

    Features (in order): see PENN_TREEBANK_TAGS for the full ordered list.
    Any tag not in PENN_TREEBANK_TAGS (rare/unexpected) is silently ignored.

    WARNING for NB users: these are non-negative counts, compatible with
    MultinomialNB. But if you combine them with VADER/AFINN (which can be
    negative), use GaussianNB or ComplementNB instead.
    """
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    counts = {t: 0 for t in PENN_TREEBANK_TAGS}
    for _, tag in tags:
        if tag in counts:
            counts[tag] += 1
    return [float(counts[t]) for t in PENN_TREEBANK_TAGS]


def compute_pos_tag_sequence(text: str) -> tuple:
    """
    Returns per-token POS tags as a list of strings.
    Used by sequence models (LSTM/RNN) where POS tags are injected
    per-token alongside word embeddings.

    Returns:
        tokens  : list of word strings, e.g. ['I', 'love', 'this']
        pos_tags: list of POS tag strings, e.g. ['PRP', 'VBP', 'DT']

    The implementor is responsible for encoding POS tags into embeddings
    (e.g. a learned POS embedding table) and concatenating/adding them
    to word embeddings at each timestep.
    """
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return tokens, [tag for _, tag in tagged]


def _sanitize_tag(tag: str) -> str:
    return tag.lower().replace('$', 'dollar').replace("''", 'close_quote').replace('``', 'open_quote').replace(',', 'comma').replace('.', 'period').replace(':', 'colon').replace('(', 'lparen').replace(')', 'rparen').replace('#', 'hash')

VADER_FEATURE_NAMES = ['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']
AFINN_FEATURE_NAMES = ['afinn_total', 'afinn_pos_count', 'afinn_neg_count', 'afinn_pos_total', 'afinn_neg_total']
POS_BROAD_FEATURE_NAMES = [f'pos_{c.lower()}_count' for c in POS_CATEGORIES]
POS_SPECIFIC_FEATURE_NAMES = [f'pos_{_sanitize_tag(t)}_count' for t in PENN_TREEBANK_TAGS]
