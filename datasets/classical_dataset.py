"""
Dataset class for classical models: Naive Bayes and Logistic Regression.

Features included:
    - TF-IDF               : always included. Fit on train split only, applied to val/test.
    - VADER                : 4 tweet-level scores (compound, pos, neg, neu). Optional.
    - AFINN                : 5 tweet-level features (total score, positive word count,
                             negative word count, positive total score, negative total score). Optional.
    - POS broad counts     : 7 broad category counts per tweet (NN, VB, JJ, RB, PR, DT, OTHER). Optional.
    - POS specific counts  : 40 Penn Treebank tag counts per tweet (finer-grained alternative). Optional.

Output format:
    scipy sparse matrix, ready to pass directly into sklearn.

NB-specific note:
    - TF-IDF only                        → use MultinomialNB (non-negative values)
    - TF-IDF + POS broad/specific counts → use MultinomialNB
    - Any VADER/AFINN added              → use GaussianNB or ComplementNB (scores can be negative)

Usage example:
    ds = ClassicalDataset(
        train_csv='data/output_file.csv',
        test_csv='data/TSAD/test.csv',
        val_size=0.1,
        use_vader=True,
        use_afinn=False,
        use_pos_broad_counts=True,
        use_pos_specific_counts=False,
        tfidf_max_features=10000,
        tfidf_ngram_range=(1, 2),
        text_col='cleaned_text',
    )
    X_train, y_train = ds.get_train()
    X_val,   y_val   = ds.get_val()
    X_test,  y_test  = ds.get_test()
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from feature_utils import (
    AFINN_FEATURE_NAMES,
    LABEL_MAP,
    POS_BROAD_FEATURE_NAMES,
    POS_SPECIFIC_FEATURE_NAMES,
    VADER_FEATURE_NAMES,
    compute_afinn_features,
    compute_pos_broad_category_counts,
    compute_pos_specific_tag_counts,
    compute_vader_features,
)


class ClassicalDataset:
    """
    Prepares features for Naive Bayes and Logistic Regression.

    Parameters
    ----------
    train_csv : str
        Path to BingXi's preprocessed training CSV (output_file.csv).
    test_csv : str
        Path to the raw test CSV (data/TSAD/test.csv).
    val_size : float
        Fraction of train_csv to hold out as validation set. Default 0.1.
    use_vader : bool
        Append 4 VADER tweet-level scores to the feature vector.
    use_afinn : bool
        Append 5 AFINN tweet-level features to the feature vector.
    use_pos_broad_counts : bool
        Append 7 broad POS category counts (NN, VB, JJ, RB, PR, DT, OTHER).
    use_pos_specific_counts : bool
        Append 40 Penn Treebank tag counts (finer-grained; mutually exclusive
        with use_pos_broad_counts — enable one or the other, not both).
    tfidf_max_features : int or None
        Vocabulary size cap for TF-IDF. None means unlimited.
    tfidf_ngram_range : tuple
        N-gram range for TF-IDF. (1,1) = unigrams, (1,2) = unigrams+bigrams.
    text_col : str
        Column to use as input text. Use 'cleaned_text' for BingXi's
        preprocessed text, or 'text' for raw tweets.
    random_state : int
        Random seed for the train/val split.

    WARNING: 
    use_pos_broad_counts and use_pos_specific_counts can 
    technically both be enabled at once, 
    but that would double-count the same POS information 
    at different granularities which is likely bad for NB
    — we recommend you pick one or the other. 

    """

    def __init__(
        self,
        train_csv: str,
        test_csv: str,
        val_size: float = 0.1,
        use_vader: bool = True,
        use_afinn: bool = False,
        use_pos_broad_counts: bool = False,
        use_pos_specific_counts: bool = False,
        tfidf_max_features: int = 10000,
        tfidf_ngram_range: tuple = (1, 2),
        text_col: str = 'cleaned_text',
        random_state: int = 42,
    ):
        self.text_col = text_col
        self.use_vader = use_vader
        self.use_afinn = use_afinn
        self.use_pos_broad_counts = use_pos_broad_counts
        self.use_pos_specific_counts = use_pos_specific_counts

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        # --- Train / val split (stratified by sentiment) ---
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            stratify=train_df['sentiment'],
            random_state=random_state,
        )

        # --- Fit TF-IDF on train only ---
        self._tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
        )
        self._tfidf.fit(train_df[text_col].fillna(''))

        # --- Build feature matrices ---
        self._X_train, self._y_train = self._build(train_df)
        self._X_val,   self._y_val   = self._build(val_df)
        self._X_test,  self._y_test  = self._build(test_df)

    def _build(self, df: pd.DataFrame):
        texts = df[self.text_col].fillna('').tolist()
        labels = np.array([LABEL_MAP[s] for s in df['sentiment']])

        # TF-IDF (sparse)
        X = self._tfidf.transform(texts)

        # Handcrafted features (dense), appended column-wise
        dense_parts = []

        if self.use_vader:
            vader_feats = np.array([compute_vader_features(t) for t in texts])
            dense_parts.append(vader_feats)

        if self.use_afinn:
            afinn_feats = np.array([compute_afinn_features(t) for t in texts])
            dense_parts.append(afinn_feats)

        if self.use_pos_broad_counts:
            pos_feats = np.array([compute_pos_broad_category_counts(t) for t in texts])
            dense_parts.append(pos_feats)

        if self.use_pos_specific_counts:
            pos_feats = np.array([compute_pos_specific_tag_counts(t) for t in texts])
            dense_parts.append(pos_feats)

        if dense_parts:
            dense = np.hstack(dense_parts)
            # Combine sparse TF-IDF with dense handcrafted features
            X = sp.hstack([X, sp.csr_matrix(dense)])

        return X, labels

    def get_train(self):
        """Returns (X_train, y_train). Pass X_train directly into sklearn .fit()."""
        return self._X_train, self._y_train

    def get_val(self):
        """Returns (X_val, y_val). Use for hyperparameter tuning."""
        return self._X_val, self._y_val

    def get_test(self):
        """Returns (X_test, y_test). Only use for final evaluation."""
        return self._X_test, self._y_test

    @property
    def feature_names(self) -> list:
        """Returns ordered list of feature names for interpretability."""
        names = self._tfidf.get_feature_names_out().tolist()
        if self.use_vader:
            names += VADER_FEATURE_NAMES
        if self.use_afinn:
            names += AFINN_FEATURE_NAMES
        if self.use_pos_broad_counts:
            names += POS_BROAD_FEATURE_NAMES
        if self.use_pos_specific_counts:
            names += POS_SPECIFIC_FEATURE_NAMES
        return names
