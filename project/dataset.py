"""
Dataset loading and PyTorch Dataset wrapper.

Supports any text + label dataset for 3-way sentiment classification.

Data sources (tried in order by default):
  1. Local CSV in project/data/
  2. Kaggle  (abhi8923shriv/sentiment-analysis-dataset)
  3. HuggingFace dataset (tweet_eval/sentiment by default)
  4. Synthetic 500-sample fallback

Labels: 0=negative, 1=neutral, 2=positive throughout.
"""

import ast
import os
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict

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


def records_from_cached_payload(payload: dict) -> List[Dict]:
    """Reconstruct record-style dicts from a Phase 1 cached payload."""
    texts = payload.get("texts") or [""] * len(payload["labels"])
    labels = payload["labels"]
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    selected_texts = payload.get("selected_texts") or [None] * len(labels)
    time_of_tweet = payload.get("time_of_tweet") or [None] * len(labels)
    age_of_user = payload.get("age_of_user") or [None] * len(labels)
    country = payload.get("country") or [None] * len(labels)
    entities = payload.get("entities") or [None] * len(labels)
    cleaned_tokens = payload.get("cleaned_tokens") or [None] * len(labels)

    records = []
    for idx, label in enumerate(labels):
        records.append(
            {
                "text": texts[idx],
                "label": int(label),
                "selected_text": selected_texts[idx],
                "time_of_tweet": time_of_tweet[idx],
                "age_of_user": age_of_user[idx],
                "country": country[idx],
                "entity": entities[idx],
                "cleaned_tokens": cleaned_tokens[idx],
            }
        )
    return records


def build_transformer_text_views(
    records: List[Dict],
    input_mode: str = "text",
    use_time_of_tweet: bool = False,
    use_age_of_user: bool = False,
    use_country: bool = False,
) -> tuple[List[str], List[int], List[str] | None]:
    """Build primary and optional secondary text views for transformer training."""
    if input_mode not in {"text", "text_plus_selected", "text_selected_pair"}:
        raise ValueError("input_mode must be 'text', 'text_plus_selected', or 'text_selected_pair'")

    texts: List[str] = []
    labels: List[int] = []
    secondary_texts: List[str] | None = [] if input_mode == "text_selected_pair" else None

    for record in records:
        segments = []
        if use_time_of_tweet and _normalize_optional_text(record.get("time_of_tweet")):
            segments.append(f"time of tweet: {record['time_of_tweet']}.")
        if use_age_of_user and _normalize_optional_text(record.get("age_of_user")):
            segments.append(f"age of user: {record['age_of_user']}.")
        if use_country and _normalize_optional_text(record.get("country")):
            segments.append(f"country: {record['country']}.")

        text = record["text"]
        selected = _normalize_optional_text(record.get("selected_text"))

        if input_mode == "text_plus_selected" and selected and selected != text:
            text = f"text: {text} sentiment span: {selected}"
        elif input_mode == "text_selected_pair":
            text = f"text: {text}"
            assert secondary_texts is not None
            secondary_texts.append(selected or record["text"])

        if segments:
            text = " ".join(segments + [text])

        texts.append(text)
        labels.append(int(record["label"]))

    return texts, labels, secondary_texts


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



def _find_local_csv():
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
# Load raw text data (local CSV / Kaggle / HuggingFace / synthetic)
# ---------------------------------------------------------------------------
def load_records(
    dataset_name: str = "tweet_eval",
    dataset_config: str = "sentiment",
    source: str = "auto",
):
    """
    Returns (train_records, val_records, test_records).

    Each record always contains:
      text, label

    Optional fields are kept when available:
      entity, cleaned_tokens, selected_text, time_of_tweet, age_of_user, country

    Args:
        dataset_name:   HuggingFace dataset name (used when source="auto" or "huggingface")
        dataset_config: HuggingFace dataset config
        source:         "auto" | "local" | "kaggle" | "huggingface" | file path
    """
    # --- Direct file path source ---
    if source not in ("auto", "local", "kaggle", "huggingface"):
        if os.path.exists(source):
            import pandas as pd
            print(f"Loading dataset from '{source}' ...")
            df = pd.read_csv(source)
            records = _records_from_dataframe(df, source_name=os.path.basename(source))
            return _split_records(records)
        else:
            raise FileNotFoundError(f"Dataset source not found: {source}")

    # --- Try 0: local cleaned CSV in project/data/ ---
    if source in ("auto", "local"):
        local_path = _find_local_csv()
        if local_path is not None:
            try:
                import pandas as pd
                print(f"Loading local dataset from '{local_path}' ...")
                df = None
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        df = pd.read_csv(local_path, encoding=encoding)
                        print(f"  Loaded with encoding={encoding}")
                        break
                    except Exception as enc_err:
                        err_str = str(enc_err).lower()
                        if "codec" in err_str or "decode" in err_str or "utf" in err_str:
                            print(f"  encoding={encoding} failed, trying next ...")
                            continue
                        raise
                if df is None:
                    raise RuntimeError("All encodings failed for local CSV")
                records = _records_from_dataframe(df, source_name="Local dataset")
                return _split_records(records)
            except Exception as e:
                print(f"  Local dataset load failed: {e}")
        if source == "local":
            raise FileNotFoundError("No local CSV found in project/data/")

    # --- Try 1: Kaggle TSAD (try multiple encodings) ---
    if source in ("auto", "kaggle"):
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
        if source == "kaggle":
            raise RuntimeError("Kaggle TSAD load failed")

    # --- Try 2: HuggingFace dataset ---
    if source in ("auto", "huggingface"):
        try:
            from datasets import load_dataset
            print(f"Loading dataset '{dataset_name}' config='{dataset_config}' ...")
            ds = load_dataset(dataset_name, dataset_config)

            def extract(split):
                return [{"text": ex["text"], "label": int(ex["label"])} for ex in ds[split]]

            train_records = extract("train")
            val_records = extract("validation")
            test_records = extract("test")

            print(
                f"  train={len(train_records)} | val={len(val_records)} | test={len(test_records)}"
            )
            return train_records, val_records, test_records
        except Exception as e:
            print(f"  HuggingFace load failed: {e}")
        if source == "huggingface":
            raise RuntimeError(f"HuggingFace dataset '{dataset_name}/{dataset_config}' load failed")

    # --- Fallback: synthetic dataset ---
    print("Falling back to synthetic 500-sample dataset.")
    tr_t, tr_l, va_t, va_l, te_t, te_l = _make_synthetic_dataset()
    return (
        [{"text": t, "label": int(y)} for t, y in zip(tr_t, tr_l)],
        [{"text": t, "label": int(y)} for t, y in zip(va_t, va_l)],
        [{"text": t, "label": int(y)} for t, y in zip(te_t, te_l)],
    )


def _make_synthetic_dataset():
    """Creates a minimal synthetic dataset for 3-way sentiment classification."""
    random.seed(42)

    templates = {
        0: [  # negative
            "I'm really disappointed with {}.",
            "Terrible experience with {}, would not recommend.",
            "The quality of {} has declined significantly.",
            "{} completely failed to meet expectations.",
            "Very frustrated with {} after this experience.",
        ],
        1: [  # neutral
            "{} released an update today.",
            "Here is some information about {}.",
            "The spokesperson for {} commented on the situation.",
            "{} announced routine changes this quarter.",
            "Analysts are monitoring {} for further developments.",
        ],
        2: [  # positive
            "{} exceeded all my expectations!",
            "Absolutely love what {} has done recently.",
            "{} delivered outstanding results this time.",
            "Really impressed with the quality of {}.",
            "Great news from {} — well deserved recognition.",
        ],
    }
    subjects = [
        "the product", "the service", "the team", "the update",
        "the platform", "the release", "the company", "the project",
    ]

    texts, labels = [], []
    n_per_class = 167  # ~500 total
    for label, tmpl_list in templates.items():
        for i in range(n_per_class):
            subject = subjects[i % len(subjects)]
            tmpl = tmpl_list[i % len(tmpl_list)]
            text = tmpl.format(subject)
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
class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        secondary_texts: List[str] | None = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.secondary_texts = secondary_texts

        if self.secondary_texts is not None and len(self.secondary_texts) != len(self.texts):
            raise ValueError("secondary_texts must have the same length as texts")

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
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": self.labels[idx],
            "text": self.texts[idx],
        }
        if self.secondary_texts is not None:
            secondary = self.tokenizer(
                self.secondary_texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            item["selected_input_ids"] = secondary["input_ids"]
            item["selected_attention_mask"] = secondary["attention_mask"]
            item["selected_text"] = self.secondary_texts[idx]
        return item


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    def _pad_sequences(sequences, fill_value):
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded.append(seq + [fill_value] * pad_len)
        return torch.tensor(padded, dtype=torch.long)

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

    collated = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attn_list, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts,
    }

    if "selected_input_ids" in batch[0]:
        collated["selected_input_ids"] = _pad_sequences(
            [b["selected_input_ids"] for b in batch],
            fill_value=pad_token_id,
        )
        collated["selected_attention_mask"] = _pad_sequences(
            [b["selected_attention_mask"] for b in batch],
            fill_value=0,
        )
        collated["selected_texts"] = [b.get("selected_text", "") for b in batch]

    return collated


def make_collate_fn(pad_token_id: int):
    def fn(batch):
        return collate_fn(batch, pad_token_id=pad_token_id)
    return fn
