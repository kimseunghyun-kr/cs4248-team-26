"""
Dataset loading and PyTorch Dataset wrapper.

Primary tweet data:
  1. TSAD via Kaggle  (abhi8923shriv/sentiment-analysis-dataset)
  2. tweet_eval/sentiment from HuggingFace
  3. Synthetic 500-sample fallback

Formal financial sentences:
  1. financial_phrasebank (sentences_allagree) from HuggingFace
  2. Synthetic template-based fallback

Labels: 0=negative, 1=neutral, 2=positive throughout.
"""

import ast
import os
import random
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from transformers import PreTrainedTokenizer


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOCAL_TWEET_FILES = [
    "output_file.csv",
    "train.csv",
    "tweet_data.csv",
]


def _find_first_column(columns, candidates):
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def _map_label(v):
    label_map = {
        "negative": 0, "Negative": 0, "-1": 0, -1: 0,
        "neutral":  1, "Neutral":  1,  "0": 1,  0: 1,
        "positive": 2, "Positive": 2,  "1": 2,  1: 2,
    }
    if v in label_map:
        return label_map[v]
    try:
        v_str = str(v).strip().lower()
        if "neg" in v_str:
            return 0
        if "neu" in v_str:
            return 1
        if "pos" in v_str:
            return 2
    except Exception:
        pass
    return None


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


def _parse_cleaned_tokens(value):
    text = _normalize_optional_text(value)
    if text is None:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(tok) for tok in parsed]
    except Exception:
        pass
    return None


def _records_from_dataframe(df, source_name: str):
    text_col = _find_first_column(
        df.columns,
        ["text", "Text", "tweet", "Tweet", "sentence", "Sentence", "content"],
    )
    label_col = _find_first_column(
        df.columns,
        ["sentiment", "Sentiment", "label", "Label", "polarity", "Polarity"],
    )
    if text_col is None or label_col is None:
        raise ValueError(
            f"{source_name}: cannot find text/label columns. Columns: {list(df.columns)}"
        )

    entity_col = _find_first_column(df.columns, ["entity", "Entity"])
    cleaned_col = _find_first_column(df.columns, ["cleaned_text", "clean_text", "tokens"])
    selected_col = _find_first_column(df.columns, ["selected_text", "Selected_text"])
    time_col = _find_first_column(df.columns, ["Time of Tweet", "time_of_tweet", "time"])
    age_col = _find_first_column(df.columns, ["Age of User", "age_of_user", "age"])
    country_col = _find_first_column(df.columns, ["Country", "country"])

    keep_cols = [text_col, label_col]
    for col in [entity_col, cleaned_col, selected_col, time_col, age_col, country_col]:
        if col is not None:
            keep_cols.append(col)
    keep_cols = list(dict.fromkeys(keep_cols))

    print(f"  {source_name}: using text_col='{text_col}', label_col='{label_col}'")
    if entity_col:
        print(f"  {source_name}: entity_col='{entity_col}'")
    if cleaned_col:
        print(f"  {source_name}: cleaned_text_col='{cleaned_col}'")
    if selected_col:
        print(f"  {source_name}: selected_text_col='{selected_col}'")

    df = df[keep_cols].dropna(subset=[text_col, label_col]).copy()
    df["_label"] = df[label_col].apply(_map_label)
    df = df.dropna(subset=["_label"])
    df["_label"] = df["_label"].astype(int)

    records = []
    for row_dict in df.to_dict(orient="records"):
        record = {
            "text": str(row_dict[text_col]),
            "label": int(row_dict["_label"]),
        }
        if entity_col is not None:
            record["entity"] = _normalize_optional_text(row_dict.get(entity_col))
        if cleaned_col is not None:
            record["cleaned_tokens"] = _parse_cleaned_tokens(row_dict.get(cleaned_col))
        if selected_col is not None:
            record["selected_text"] = _normalize_optional_text(row_dict.get(selected_col))
        if time_col is not None:
            record["time_of_tweet"] = _normalize_optional_text(row_dict.get(time_col))
        if age_col is not None:
            record["age_of_user"] = _normalize_optional_text(row_dict.get(age_col))
        if country_col is not None:
            record["country"] = _normalize_optional_text(row_dict.get(country_col))
        records.append(record)

    labels = [r["label"] for r in records]
    print(f"  {source_name} loaded: {len(records)} samples | "
          f"neg={labels.count(0)} neu={labels.count(1)} pos={labels.count(2)}")
    return records


def _split_records(records):
    from sklearn.model_selection import train_test_split

    labels = [r["label"] for r in records]
    tr, tmp = train_test_split(
        records, test_size=0.30, random_state=42, stratify=labels
    )
    tmp_labels = [r["label"] for r in tmp]
    va, te = train_test_split(
        tmp, test_size=0.50, random_state=42, stratify=tmp_labels
    )
    print(f"  split -> train={len(tr)} | val={len(va)} | test={len(te)}")
    return tr, va, te


def _records_to_legacy_tuple(train_records, val_records, test_records):
    return (
        [r["text"] for r in train_records],
        [r["label"] for r in train_records],
        [r["text"] for r in val_records],
        [r["label"] for r in val_records],
        [r["text"] for r in test_records],
        [r["label"] for r in test_records],
    )


def _find_local_tweet_csv():
    for name in LOCAL_TWEET_FILES:
        path = os.path.join(LOCAL_DATA_DIR, name)
        if os.path.exists(path):
            return path
    if not os.path.isdir(LOCAL_DATA_DIR):
        return None
    for name in sorted(os.listdir(LOCAL_DATA_DIR)):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(LOCAL_DATA_DIR, name)
        try:
            import pandas as pd
            cols = pd.read_csv(path, nrows=1).columns.tolist()
        except Exception:
            continue
        if _find_first_column(cols, ["text", "Text", "tweet", "Tweet", "sentence", "Sentence", "content"]) and \
           _find_first_column(cols, ["sentiment", "Sentiment", "label", "Label", "polarity", "Polarity"]):
            return path
    return None


# ---------------------------------------------------------------------------
# Load raw tweet data (TSAD / tweet_eval / synthetic)
# ---------------------------------------------------------------------------
def load_tsad_records(dataset_name: str = "tweet_eval", dataset_config: str = "sentiment"):
    """
    Returns (train_records, val_records, test_records).

    Each record always contains:
      text, label

    Optional fields are kept when available:
      entity, cleaned_tokens, selected_text, time_of_tweet, age_of_user, country
    """
    # --- Try 0: local cleaned CSV in project/data/ ----------------------------
    # local_path = _find_local_tweet_csv()
    # if local_path is not None:
    #     try:
    #         import pandas as pd
    #         print(f"Loading local tweet dataset from '{local_path}' ...")
    #         df = None
    #         for encoding in ["utf-8", "latin-1", "cp1252"]:
    #             try:
    #                 df = pd.read_csv(local_path, encoding=encoding)
    #                 print(f"  Loaded with encoding={encoding}")
    #                 break
    #             except Exception as enc_err:
    #                 err_str = str(enc_err).lower()
    #                 if "codec" in err_str or "decode" in err_str or "utf" in err_str:
    #                     print(f"  encoding={encoding} failed, trying next ...")
    #                     continue
    #                 raise
    #         if df is None:
    #             raise RuntimeError("All encodings failed for local tweet CSV")
    #         records = _records_from_dataframe(df, source_name="Local tweet dataset")
    #         return _split_records(records)
    #     except Exception as e:
    #         print(f"  Local dataset load failed: {e}")

    # --- Try 1: TSAD from Kaggle (try multiple encodings) ---------------------
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        print("Loading TSAD from Kaggle (abhi8923shriv/sentiment-analysis-dataset) ...")
        df = None
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    "abhi8923shriv/sentiment-analysis-dataset",
                    "train.csv",
                    pandas_kwargs={"encoding": encoding},
                )
                print(f"  Loaded with encoding={encoding}")
                break
            except Exception as enc_err:
                err_str = str(enc_err).lower()
                if "codec" in err_str or "decode" in err_str or "utf" in err_str:
                    print(f"  encoding={encoding} failed, trying next ...")
                    continue
                raise  # non-encoding error — propagate immediately
        if df is None:
            raise RuntimeError("All encodings failed for TSAD CSV")
        records = _records_from_dataframe(df, source_name="TSAD")
        return _split_records(records)
    except Exception as e:
        print(f"  Kaggle load failed: {e}")

    # --- Try 2: tweet_eval from HuggingFace -----------------------------------
    # try:
    #     from datasets import load_dataset
    #     print(f"Loading dataset '{dataset_name}' config='{dataset_config}' ...")
    #     ds = load_dataset(dataset_name, dataset_config)

    #     def extract(split):
    #         return [{"text": ex["text"], "label": int(ex["label"])} for ex in ds[split]]

    #     train_records = extract("train")
    #     val_records = extract("validation")
    #     test_records = extract("test")

    #     print(
    #         f"  train={len(train_records)} | val={len(val_records)} | test={len(test_records)}"
    #     )
    #     return train_records, val_records, test_records

    # except Exception as e:
    #     print(f"WARNING: Could not load '{dataset_name}/{dataset_config}': {e}")
    #     print("Falling back to synthetic 500-sample dataset.")
    #     tr_t, tr_l, va_t, va_l, te_t, te_l = _make_synthetic_dataset()
    #     return (
    #         [{"text": t, "label": int(y)} for t, y in zip(tr_t, tr_l)],
    #         [{"text": t, "label": int(y)} for t, y in zip(va_t, va_l)],
    #         [{"text": t, "label": int(y)} for t, y in zip(te_t, te_l)],
    #     )


def load_tsad(dataset_name: str = "tweet_eval", dataset_config: str = "sentiment"):
    """
    Returns (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels).
    Labels: 0=negative, 1=neutral, 2=positive.
    """
    return _records_to_legacy_tuple(*load_tsad_records(dataset_name, dataset_config))


def _parse_tsad_kaggle(df):
    """
    Parse the TSAD Kaggle dataframe.

    Expected columns (auto-detected):
      text column  : 'text', 'Text', 'tweet', 'Tweet', 'sentence'
      label column : 'sentiment', 'Sentiment', 'label', 'Label', 'polarity'

    Label values → 0/1/2:
      'negative' / 'Negative' / '-1' / 0  → 0
      'neutral'  / 'Neutral'  / '0'  / 1  → 1
      'positive' / 'Positive' / '1'  / 2  → 2
    """
    records = _records_from_dataframe(df, source_name="TSAD")
    return _records_to_legacy_tuple(*_split_records(records))




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
