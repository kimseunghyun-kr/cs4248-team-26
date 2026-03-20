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
    from datasets import load_dataset

    # Candidate datasets in Parquet format (no loading scripts required).
    # Each entry: (dataset_id, config_or_None, text_field)
    candidates = [
        ("nickmuchi/financial-classification", None,                   "text"),
        ("FinanceInc/auditor_sentiment",       None,                   "sentence"),
        ("Dogeek/financial_phrasebank",        "sentences_allagree",   "sentence"),
        ("Satarupa/financial_phrasebank",      "sentences_allagree",   "sentence"),
    ]

    for dataset_id, config, text_field in candidates:
        try:
            print(f"Loading {dataset_id} ...")
            ds = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)
            split = "train" if "train" in ds else list(ds.keys())[0]
            texts = [ex[text_field] for ex in ds[split] if ex.get(text_field)]
            if len(texts) > 100:
                print(f"  {dataset_id}: {len(texts)} formal sentences")
                return texts
        except Exception as e:
            print(f"  {dataset_id} failed: {e}")

    # --- Fallback: handcrafted FPB-style financial news sentences -------------
    print("  Using handcrafted FPB-style formal sentences as fallback.")
    return _fpb_style_sentences()


def _fpb_style_sentences() -> List[str]:
    """
    Handcrafted sentences written in the register of Financial PhraseBank
    (Reuters financial news, short factual statements).
    These represent the distribution FinBERT was fine-tuned on.
    """
    return [
        # --- Positive sentiment ---
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the year-earlier period.",
        "The company's net sales increased by 9.5 percent to EUR 562.5 million.",
        "Net profit for the period climbed to EUR 37.6 million from EUR 21.3 million a year ago.",
        "The board proposed a dividend of EUR 0.22 per share, up from EUR 0.18 the previous year.",
        "Revenue grew 11 percent year-on-year to reach EUR 1.04 billion.",
        "The firm reported a record operating margin of 14.2 percent for the quarter.",
        "Earnings per share rose to EUR 1.43, beating analyst expectations of EUR 1.31.",
        "The company raised its full-year guidance following stronger-than-expected second-quarter results.",
        "Order intake increased 18 percent compared with the same period last year.",
        "The group's EBITDA improved to EUR 88 million from EUR 71 million.",
        "Shares in the company rose 6.2 percent after the earnings announcement.",
        "The acquisition is expected to be earnings accretive from the first full year of consolidation.",
        "Cash flow from operations increased to EUR 142 million, up from EUR 98 million previously.",
        "The company secured a EUR 320 million contract with a major European utility provider.",
        "Return on equity improved to 17.4 percent from 13.8 percent a year earlier.",
        "The firm's market share in the Nordic region expanded to 34 percent.",
        "Comparable sales growth reached 7.3 percent, driven by strong performance in Asia-Pacific.",
        "The company completed the divestiture of its non-core logistics unit for EUR 415 million.",
        "Net interest income rose 8 percent to EUR 1.2 billion in the first half.",
        "The group announced a EUR 200 million share buyback programme.",
        "Personnel costs declined as a proportion of revenue, reflecting improved operational efficiency.",
        "The company's credit rating was upgraded to A- by Standard & Poor's.",
        "Gross margin improved by 1.8 percentage points to 42.6 percent.",
        "The firm reported its fifth consecutive quarter of double-digit revenue growth.",
        "Loan portfolio quality improved with non-performing loans declining to 1.8 percent.",
        # --- Negative sentiment ---
        "Operating loss widened to EUR 12.3 million from EUR 4.7 million a year ago.",
        "The company lowered its full-year sales forecast citing weaker demand in Europe.",
        "Net loss for the period amounted to EUR 28.4 million compared with a profit of EUR 6.2 million.",
        "The firm announced plans to cut 1,200 jobs as part of a restructuring programme.",
        "Revenue fell 7.4 percent to EUR 381 million, missing the consensus estimate of EUR 412 million.",
        "The company filed for creditor protection following a deterioration in liquidity.",
        "Impairment charges of EUR 95 million were recorded on goodwill related to the 2019 acquisition.",
        "The board suspended the annual dividend in response to the deteriorating financial position.",
        "Operating cash flow turned negative, declining to minus EUR 23 million in the quarter.",
        "The company warned that full-year EBIT would fall short of prior guidance by approximately 20 percent.",
        "Shares fell 11.3 percent to their lowest level in three years following the profit warning.",
        "The firm's debt-to-equity ratio rose to 2.8 times following the refinancing of its credit facility.",
        "Write-downs on inventory totalling EUR 47 million weighed on quarterly results.",
        "The company's order backlog declined by 14 percent compared with the same period last year.",
        "Market conditions in the construction segment remained challenging throughout the period.",
        "The rating agency placed the company's debt on negative credit watch.",
        "Cost overruns on a major infrastructure project led to an exceptional charge of EUR 63 million.",
        "The company reported that its largest customer had terminated a long-term supply agreement.",
        "Gross margin contracted by 3.1 percentage points due to rising raw material costs.",
        "The firm disclosed a regulatory investigation into its pricing practices in three markets.",
        # --- Neutral sentiment ---
        "The company will publish its half-year results on 14 August.",
        "Nokian Tyres said it would hold its annual general meeting on 28 March in Helsinki.",
        "The board of directors decided to maintain the dividend at EUR 0.30 per share.",
        "The firm appointed Mikko Helander as its new chief executive officer, effective 1 March.",
        "Outokumpu said it would release its interim report for the first quarter on 29 April.",
        "The company operates 47 production facilities across 18 countries.",
        "Nokia confirmed that discussions with the potential acquirer were ongoing.",
        "The group employs approximately 8,400 people in Finland and 21,000 worldwide.",
        "The company said the transaction remained subject to regulatory approval.",
        "Fortum's board proposed an unchanged dividend of EUR 1.14 per share for 2013.",
        "The interim chief financial officer will assume the role on a permanent basis pending board confirmation.",
        "The company reiterated its full-year financial targets at the investor day presentation.",
        "YIT said it would divest its industrial services division by the end of the financial year.",
        "The firm stated that it did not comment on market speculation regarding potential transactions.",
        "Kesko Corporation reported that its retail division accounted for 61 percent of group sales.",
        "The company's fiscal year ends on 31 December.",
        "UPM-Kymmene said it had completed the previously announced capacity reduction.",
        "The group operates three reportable business segments: energy, paper, and pulp.",
        "Metso said it would transfer its mining and construction equipment businesses to a new entity.",
        "The company confirmed the terms of the rights issue announced on 7 February.",
        "Elisa Corporation released its third-quarter results in line with preliminary figures.",
        "The supervisory board approved the proposed amendments to the articles of association.",
        "Wärtsilä said the acquisition had been completed following receipt of all required regulatory approvals.",
        "The annual report is available on the company's investor relations website.",
        "The company's shares are listed on the Nasdaq Helsinki exchange.",
        "Stora Enso said it would invest EUR 170 million in its packaging board mill in Imatra.",
        "The chief executive stated that the business environment remained uncertain.",
        "The company has not yet determined the size or timing of any potential capital markets transaction.",
        "Neste Oil said it would continue to evaluate strategic options for its retail network.",
        "The board noted that the results were broadly in line with management expectations.",
        "The company said operating conditions in its main markets showed little change from the prior quarter.",
        "Sanoma Corporation confirmed that the divestiture process was proceeding according to plan.",
        "The firm stated that its financial position remained solid with adequate liquidity reserves.",
        "The annual general meeting approved all items on the agenda as proposed by the board.",
        "Tieto said its services segment had been reorganised into four business lines.",
        "The company maintained its outlook for the full financial year.",
        "Talvivaara Mining Company said it had received the necessary environmental permits.",
        "Stockmann noted that the weak consumer sentiment had persisted into the second half.",
        "The company issued a stock exchange release correcting an error in its previous announcement.",
        "Cargotec said the Board of Directors had authorised the company to repurchase its own shares.",
        "The firm confirmed that there had been no material changes to its financial position since the last report.",
        "Ramirent's board decided to convene an extraordinary general meeting on 5 December.",
        "The company disclosed that it had received an offer for one of its business units.",
        "F-Secure said it would focus its product portfolio on cybersecurity solutions for enterprises.",
        "The board of Fiskars resolved to distribute a dividend of EUR 0.68 per share.",
        "Huhtamäki said the new production line had commenced operations as scheduled.",
        "The group's net debt stood at EUR 1.3 billion at the end of the reporting period.",
        "Cramo said the integration of the acquired businesses was progressing according to plan.",
        "The company noted that currency fluctuations had a limited impact on reported results.",
    ]


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
