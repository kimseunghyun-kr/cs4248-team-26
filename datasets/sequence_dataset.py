"""
Dataset class for sequence models: LSTM and RNN.

Features included:
    - Tokens      : list of word strings per tweet. YOU feed these into your
                    embedding layer (Word2Vec, GloVe, FastText, etc.) to get
                    tensors of shape [seq_len, embed_dim].
    - POS tags    : per-token POS tag strings (optional). YOU encode these
                    into a learned POS embedding table and add/concatenate
                    to word embeddings at each timestep.
    - Tweet-level : VADER and/or AFINN scores (optional). Concatenate these
                    to the final hidden state h_T before your classifier head.

Where to plug in embeddings:
    ┌─────────────────────────────────────────────────────────────┐
    │  tokens  ──► YOUR embedding layer ──► [seq_len, embed_dim]  │
    │  pos_tags ──► YOUR POS embed table ──► [seq_len, pos_dim]   │
    │     └─ concatenate/add ──► LSTM input [seq_len, embed_dim]  │
    │                                                             │
    │  LSTM output: h_T  [batch, hidden_dim]                      │
    │  tweet_features ──► concatenate to h_T                      │
    │     └─► classifier head                                     │
    └─────────────────────────────────────────────────────────────┘

Usage example:
    from torch.utils.data import DataLoader

    train_ds = SequenceDataset(
        csv_path='data/output_train.csv',
        split='train',
        val_size=0.1,
        use_vader=True,
        use_afinn=False,
        use_pos_tags=True,
        text_col='cleaned_text',
    )
    loader = DataLoader(train_ds, batch_size=32, collate_fn=train_ds.collate_fn)

    for tokens, pos_tags, tweet_feats, labels in loader:
        # tokens     : list of lists of strings  (batch_size, seq_len)
        # pos_tags   : list of lists of strings  (batch_size, seq_len) or None
        # tweet_feats: FloatTensor               (batch_size, n_tweet_features)
        # labels     : LongTensor                (batch_size,)

        # --- YOUR embedding step (goes in your model, not here) ---
        # word_embeds = your_embed_layer(tokens)        # [batch, seq_len, embed_dim]
        # pos_embeds  = your_pos_embed_layer(pos_tags)  # [batch, seq_len, pos_dim]
        # lstm_input  = torch.cat([word_embeds, pos_embeds], dim=-1)
        # h_T, _      = your_lstm(lstm_input)
        # h_T_final   = torch.cat([h_T, tweet_feats], dim=-1)
        # logits      = your_classifier(h_T_final)
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
    compute_pos_tag_sequence,
)


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for LSTM/RNN models.

    Parameters
    ----------
    csv_path : str
        Path to BingXi's preprocessed CSV.
    split : str
        One of 'train', 'val', or 'test'. Determines which subset to return.
        'train' and 'val' are carved from csv_path. For 'test', provide
        test_csv_path instead.
    test_csv_path : str or None
        Path to the test CSV. Required when split='test'.
    val_size : float
        Fraction of csv_path to hold out as validation. Default 0.1.
    use_vader : bool
        Include 4 VADER tweet-level scores as tweet_features.
    use_afinn : bool
        Include 5 AFINN tweet-level features as tweet_features.
    use_pos_tags : bool
        Return per-token POS specifictag strings alongside tokens. (not broad category)
    text_col : str
        Column to use as input text. Default 'cleaned_text'.
    random_state : int
        Seed for train/val split.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        test_csv_path: str = None,
        val_size: float = 0.1,
        use_vader: bool = True,
        use_afinn: bool = False,
        use_pos_tags: bool = False,
        text_col: str = 'cleaned_text',
        random_state: int = 42,
    ):
        assert split in ('train', 'val', 'test'), "split must be 'train', 'val', or 'test'"

        self.use_pos_tags = use_pos_tags
        self.use_vader = use_vader
        self.use_afinn = use_afinn

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

        # --- Tokenise and optionally POS-tag ---
        if use_pos_tags:
            from feature_utils import compute_pos_tag_sequence
            parsed = [compute_pos_tag_sequence(t) for t in texts]
            self.tokens   = [p[0] for p in parsed]
            self.pos_tags = [p[1] for p in parsed]
        else:
            self.tokens   = [t.split() for t in texts]
            self.pos_tags = None

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
            self.tweet_features = np.hstack(tweet_feat_parts)  # [N, n_tweet_feats]
        else:
            self.tweet_features = np.zeros((len(texts), 0), dtype=np.float32)

        self.labels = np.array([LABEL_MAP[s] for s in df['sentiment']], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns
        -------
        tokens       : list of str — feed into YOUR embedding layer
        pos_tags     : list of str or None — feed into YOUR POS embedding table
        tweet_feats  : FloatTensor [n_tweet_features] — concat to h_T
        label        : LongTensor scalar
        """
        return (
            self.tokens[idx],
            self.pos_tags[idx] if self.pos_tags is not None else None,
            torch.tensor(self.tweet_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.
        Keeps tokens and pos_tags as lists of lists (variable length).
        Pass this as collate_fn= to DataLoader.
        """
        tokens_batch    = [item[0] for item in batch]
        pos_tags_batch  = [item[1] for item in batch]
        tweet_feats     = torch.stack([item[2] for item in batch])
        labels          = torch.stack([item[3] for item in batch])

        pos_tags_batch = pos_tags_batch if pos_tags_batch[0] is not None else None
        return tokens_batch, pos_tags_batch, tweet_feats, labels

    @property
    def tweet_feature_dim(self) -> int:
        """Dimension of the tweet-level feature vector. Used to size your model."""
        return self.tweet_features.shape[1]
