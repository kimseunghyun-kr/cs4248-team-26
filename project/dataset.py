"""
Dataset loading and PyTorch Dataset wrapper.

Primary: tweet_eval / sentiment  (0=negative, 1=neutral, 2=positive)
Fallback: synthetic 500-sample dataset (noted in output)
"""

import random
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
def load_tsad(dataset_name: str = "tweet_eval", dataset_config: str = "sentiment"):
    """
    Returns (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels).
    Labels: 0=negative, 1=neutral, 2=positive.
    """
    try:
        from datasets import load_dataset
        print(f"Loading dataset '{dataset_name}' config='{dataset_config}' ...")
        ds = load_dataset(dataset_name, dataset_config)

        def extract(split):
            texts = [ex["text"] for ex in ds[split]]
            labels = [ex["label"] for ex in ds[split]]
            return texts, labels

        train_texts, train_labels = extract("train")
        val_texts, val_labels = extract("validation")
        test_texts, test_labels = extract("test")

        print(
            f"  train={len(train_texts)} | val={len(val_texts)} | test={len(test_texts)}"
        )
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

    except Exception as e:
        print(f"WARNING: Could not load '{dataset_name}/{dataset_config}': {e}")
        print("Falling back to synthetic 500-sample dataset.")
        return _make_synthetic_dataset()


def _make_synthetic_dataset():
    """Creates a minimal synthetic financial tweet dataset."""
    random.seed(42)

    templates = {
        0: [  # negative
            "Stock {} crashed today, down {} percent.",
            "Earnings miss for {}, losses widen to ${}M.",
            "Revenue declined sharply for {} this quarter.",
            "{} faces bankruptcy fears as debt soars.",
            "Investors dump {} shares after poor guidance.",
        ],
        1: [  # neutral
            "Company {} released its quarterly report today.",
            "{} will hold its annual shareholder meeting next week.",
            "The CEO of {} commented on market conditions.",
            "{} announced a routine dividend payment.",
            "Analysts maintain hold rating on {}.",
        ],
        2: [  # positive
            "{} stock surged {}% on strong earnings.",
            "Record revenue for {} this quarter, beating estimates.",
            "{} raises full-year guidance after strong performance.",
            "Investors cheer as {} announces share buyback.",
            "{} profit rose sharply, topping analyst forecasts.",
        ],
    }
    tickers = ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "NVDA", "META", "JPM"]

    texts, labels = [], []
    n_per_class = 167  # ~500 total
    for label, tmpl_list in templates.items():
        for i in range(n_per_class):
            ticker = tickers[i % len(tickers)]
            tmpl = tmpl_list[i % len(tmpl_list)]
            val = random.randint(2, 40)
            try:
                text = tmpl.format(ticker, val)
            except IndexError:
                text = tmpl.format(ticker)
            texts.append(text)
            labels.append(label)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    n = len(texts)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_texts, train_labels = texts[:n_train], labels[:n_train]
    val_texts, val_labels = texts[n_train : n_train + n_val], labels[n_train : n_train + n_val]
    test_texts, test_labels = texts[n_train + n_val :], labels[n_train + n_val :]

    print(
        f"  Synthetic: train={len(train_texts)} | val={len(val_texts)} | test={len(test_texts)}"
    )
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": self.labels[idx],
            "text": self.texts[idx],
        }


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids_list, attn_list, labels, texts = [], [], [], []
    for b in batch:
        ids = b["input_ids"]
        mask = b["attention_mask"]
        pad_len = max_len - len(ids)
        input_ids_list.append(ids + [pad_token_id] * pad_len)
        attn_list.append(mask + [0] * pad_len)
        labels.append(b["label"])
        texts.append(b["text"])

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attn_list, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts,
    }


def make_collate_fn(pad_token_id: int):
    def fn(batch):
        return collate_fn(batch, pad_token_id=pad_token_id)
    return fn
