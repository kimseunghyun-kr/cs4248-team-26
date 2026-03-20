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

import random
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Load raw tweet data (TSAD / tweet_eval / synthetic)
# ---------------------------------------------------------------------------
def load_tsad(dataset_name: str = "tweet_eval", dataset_config: str = "sentiment"):
    """
    Returns (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels).
    Labels: 0=negative, 1=neutral, 2=positive.
    """
    # --- Try 1: TSAD from Kaggle (try multiple encodings) ----------------------
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
        return _parse_tsad_kaggle(df)
    except Exception as e:
        print(f"  Kaggle load failed: {e}")

    # --- Try 2: tweet_eval from HuggingFace ------------------------------------
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
    # Detect text column
    text_col = None
    for c in ["text", "Text", "tweet", "Tweet", "sentence", "Sentence", "content"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"Cannot find text column. Columns: {list(df.columns)}")

    # Detect label column
    label_col = None
    for c in ["sentiment", "Sentiment", "label", "Label", "polarity", "Polarity"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Cannot find label column. Columns: {list(df.columns)}")

    print(f"  TSAD: using text_col='{text_col}', label_col='{label_col}'")

    # Drop rows with NaN text or label
    df = df[[text_col, label_col]].dropna()

    # Map labels to 0/1/2
    label_map = {
        "negative": 0, "Negative": 0, "-1": 0, -1: 0,
        "neutral":  1, "Neutral":  1,  "0": 1,  0: 1,
        "positive": 2, "Positive": 2,  "1": 2,  1: 2,
    }

    def map_label(v):
        # Handle both string and numeric
        if v in label_map:
            return label_map[v]
        try:
            v_str = str(v).strip().lower()
            if "neg" in v_str:
                return 0
            elif "neu" in v_str:
                return 1
            elif "pos" in v_str:
                return 2
        except Exception:
            pass
        return None

    df["_label"] = df[label_col].apply(map_label)
    df = df.dropna(subset=["_label"])
    df["_label"] = df["_label"].astype(int)

    texts  = df[text_col].tolist()
    labels = df["_label"].tolist()

    print(f"  TSAD loaded: {len(texts)} samples | "
          f"neg={labels.count(0)} neu={labels.count(1)} pos={labels.count(2)}")

    # Stratified 70/15/15 split
    from sklearn.model_selection import train_test_split
    tr_t, tmp_t, tr_l, tmp_l = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    va_t, te_t, va_l, te_l = train_test_split(
        tmp_t, tmp_l, test_size=0.50, random_state=42, stratify=tmp_l
    )
    print(f"  split → train={len(tr_t)} | val={len(va_t)} | test={len(te_t)}")
    return tr_t, tr_l, va_t, va_l, te_t, te_l


# ---------------------------------------------------------------------------
# Load formal financial sentences (for style contrast corpus)
# ---------------------------------------------------------------------------
def load_formal_sentences() -> List[str]:
    """
    Returns a list of formal financial sentences (no labels needed).

    Sources tried in order:
      1. financial_phrasebank (sentences_allagree) from HuggingFace
      2. Synthetic template-based fallback (~2000 sentences)
    """
    # --- Try 1: takala/financial_phrasebank (Parquet mirror, no loading script) --
    try:
        from datasets import load_dataset
        print("Loading takala/financial_phrasebank (sentences_allagree) ...")
        ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
        texts = [ex["sentence"] for ex in ds["train"]]
        print(f"  financial_phrasebank: {len(texts)} formal sentences")
        return texts
    except Exception as e:
        print(f"  takala/financial_phrasebank load failed: {e}")

    # --- Try 2: original financial_phrasebank with trust_remote_code ----------
    try:
        from datasets import load_dataset
        print("Loading financial_phrasebank (trust_remote_code=True) ...")
        ds = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
        texts = [ex["sentence"] for ex in ds["train"]]
        print(f"  financial_phrasebank: {len(texts)} formal sentences")
        return texts
    except Exception as e:
        print(f"  financial_phrasebank load failed: {e}")

    # --- Fallback: synthetic formal sentences ---------------------------------
    print("  Using synthetic formal financial sentences as fallback.")
    return _make_synthetic_formal_sentences()


def _make_synthetic_formal_sentences(n: int = 2000) -> List[str]:
    """Generate diverse formal financial sentences for style contrast."""
    random.seed(42)
    tickers = ["Apple Inc.", "Alphabet Inc.", "Tesla Inc.", "Microsoft Corp.",
               "Amazon.com Inc.", "NVIDIA Corp.", "Meta Platforms Inc.", "JPMorgan Chase & Co.",
               "Berkshire Hathaway Inc.", "Johnson & Johnson"]
    sectors = ["technology", "healthcare", "energy", "financial services",
                "consumer goods", "industrials", "utilities", "materials"]
    templates = [
        "{company} reported quarterly revenue of ${amount} billion, representing a {pct}% change year-over-year.",
        "The board of directors of {company} approved a dividend of ${div} per share.",
        "{company} announced the acquisition of a {sector} firm for approximately ${amount} billion.",
        "Analysts at Goldman Sachs maintained a Buy rating on {company} with a price target of ${price}.",
        "{company} disclosed in its 10-Q filing that operating income declined by {pct}% in the third quarter.",
        "The Federal Reserve's interest rate decision is expected to affect {sector} sector valuations.",
        "{company} completed a share repurchase program, retiring {amount} million shares.",
        "Credit rating agency Moody's affirmed {company}'s investment-grade rating of Baa2.",
        "Regulatory filings indicate that {company} has increased its capital expenditure guidance for fiscal year 2025.",
        "{company} management reaffirmed full-year earnings per share guidance of ${price}.",
        "The Securities and Exchange Commission approved {company}'s proposed merger with its subsidiary.",
        "{company} issued $2.{amount} billion in senior unsecured notes due 2031 at a yield of {pct}%.",
        "Institutional ownership of {company} increased to {pct}% according to the latest 13F filings.",
        "The {sector} sector underperformed the broader market index by {pct} basis points last quarter.",
        "{company}'s gross margin expanded by {pct} basis points driven by improved supply chain efficiency.",
        "Free cash flow for {company} totaled ${amount} billion in the trailing twelve months.",
        "{company} disclosed a material weakness in its internal controls over financial reporting.",
        "The consensus earnings estimate for {company} was revised upward by analysts following strong preliminary results.",
        "{company} filed for Chapter 11 bankruptcy protection, citing elevated debt levels and declining revenues.",
        "Working capital for {company} increased to ${amount} billion as of the end of the reporting period.",
    ]
    sentences = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        company = tickers[i % len(tickers)]
        sector  = sectors[i % len(sectors)]
        amount  = round(random.uniform(0.5, 50.0), 1)
        pct     = round(random.uniform(1.0, 25.0), 1)
        div     = round(random.uniform(0.10, 2.50), 2)
        price   = round(random.uniform(50, 500), 0)
        try:
            s = tmpl.format(company=company, sector=sector, amount=amount,
                            pct=pct, div=div, price=int(price))
        except KeyError:
            s = tmpl.format(company=company, sector=sector, amount=amount, pct=pct,
                            div=div, price=int(price))
        sentences.append(s)
    return sentences


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
