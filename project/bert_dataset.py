"""
Dataset class for BERT (and other transformer models).

Features included:
    - text         : cleaned tweet string. Pass this to YOUR BERT tokenizer.
                     Do NOT tokenize here — BERT uses its own WordPiece tokenizer.
    - Tweet-level  : VADER and/or AFINN scores (optional). Concatenate these
                     to the [CLS] token representation before your classifier head.

Where to plug in:
    ┌───────────────────────────────────────────────────────────────────┐
    │  text ──► YOUR tokenizer (AutoTokenizer) ──► input_ids,          │
    │                                               attention_mask      │
    │     └─► YOUR BERT model ──► last_hidden_state[:, 0, :]  (CLS)    │
    │                                                                   │
    │  tweet_features ──► concatenate to CLS vector                    │
    │     └─► classifier head                                           │
    └───────────────────────────────────────────────────────────────────┘

Usage example:
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_ds = BERTDataset(
        csv_path='data/output_train.csv',
        split='train',
        val_size=0.1,
        use_vader=True,
        use_afinn=False,
        text_col='cleaned_text',
    )
    loader = DataLoader(train_ds, batch_size=16)

    for texts, tweet_feats, labels in loader:
        # texts       : list of str           (batch_size,)
        # tweet_feats : FloatTensor           (batch_size, n_tweet_features)
        # labels      : LongTensor            (batch_size,)

        # --- YOUR tokenization + forward pass (goes in your model/training loop) ---
        # encoding = tokenizer(texts, padding=True, truncation=True,
        #                      max_length=128, return_tensors='pt')
        # outputs  = bert_model(**encoding)
        # cls_vec  = outputs.last_hidden_state[:, 0, :]        # [batch, hidden_dim]
        # combined = torch.cat([cls_vec, tweet_feats], dim=-1)  # [batch, hidden_dim + n_feats]
        # logits   = classifier_head(combined)
        pass
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from feature_utils import (
    LABEL_MAP,
    compute_afinn_features,
    compute_vader_features,
)


class BERTDataset(Dataset):
    """
    PyTorch Dataset for BERT and other transformer models.

    Parameters
    ----------
    csv_path : str
        Path to BingXi's preprocessed CSV.
    split : str
        One of 'train', 'val', or 'test'.
    test_csv_path : str or None
        Path to the test CSV. Required when split='test'.
    val_size : float
        Fraction of csv_path to hold out as validation. Default 0.1.
    use_vader : bool
        Include 4 VADER tweet-level scores as tweet_features.
    use_afinn : bool
        Include 5 AFINN tweet-level features as tweet_features.
    text_col : str
        Column to feed into BERT. Default 'cleaned_text'.
        Switch to 'text' to give BERT raw unprocessed tweets — this is
        often better for BERT since it handles punctuation and casing itself.
    random_state : int
        Seed for train/val split.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        test_csv_path: str = None,
        val_size: float = 0.1,
        use_vader: bool = False,
        use_afinn: bool = False,
        text_col: str = 'cleaned_text',
        random_state: int = 42,
    ):
        assert split in ('train', 'val', 'test'), "split must be 'train', 'val', or 'test'"

        if split == 'test':
            assert test_csv_path is not None, "Provide test_csv_path when split='test'"
            df = pd.read_csv(test_csv_path)
        else:
            full_df = pd.read_csv(csv_path)
            train_df, val_df = train_test_split(
                full_df,
                test_size=val_size,
                stratify=full_df['sentiment'],
                random_state=random_state,
            )
            df = train_df if split == 'train' else val_df

        df = df.reset_index(drop=True)
        texts = df[text_col].fillna('').tolist()

        self.texts = texts

        # --- Tweet-level features ---
        tweet_feat_parts = []
        if use_vader:
            tweet_feat_parts.append(
                np.array([compute_vader_features(t) for t in texts], dtype=np.float32)
            )
        if use_afinn:
            tweet_feat_parts.append(
                np.array([compute_afinn_features(t) for t in texts], dtype=np.float32)
            )

        if tweet_feat_parts:
            self.tweet_features = np.hstack(tweet_feat_parts)
        else:
            self.tweet_features = np.zeros((len(texts), 0), dtype=np.float32)

        self.labels = np.array([LABEL_MAP[s] for s in df['sentiment']], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns
        -------
        text         : str — pass this to AutoTokenizer in your training loop
        tweet_feats  : FloatTensor [n_tweet_features] — concat to CLS vector
        label        : LongTensor scalar
        """
        return (
            self.texts[idx],
            torch.tensor(self.tweet_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    @property
    def tweet_feature_dim(self) -> int:
        """Dimension of the tweet-level feature vector. Used to size your classifier head."""
        return self.tweet_features.shape[1]
