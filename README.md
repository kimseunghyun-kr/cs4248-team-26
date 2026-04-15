# CS4248 Team 26 — Tweet Sentiment Classification

End-to-end tweet sentiment classification (negative / neutral / positive) progressing from classical TF-IDF baselines to fine-tuned transformers to diverse soft-vote ensembles.

**Best result:** 5-model soft-vote ensemble (twitter-roberta + bertweet + bertweet-sentiment + twitter-xlm-roberta + deberta-v3-large), test F1 macro **0.8276** (`ens_5_abcde`). See [`report.md`](report.md) for the narrative and [`report_analysis.md`](report_analysis.md) for the ablation-focused analytical report.

## Results at a glance

| Tier        | Best config                                  | Test F1    |
|-------------|----------------------------------------------|------------|
| Classical   | LightGBM + TF-IDF char+word n-grams + VADER  | 0.7514     |
| Transformer | BERTweet-base (raw text)                     | 0.8120     |
| Ensemble    | 5-model diverse soft-vote (`ens_5_abcde`)       | **0.8276** |

Full per-run numbers in `results/all_results.csv` (117 rows across all three tiers).

## Directory structure

```
cs4248-team-26/
├── README.md                  # this file
├── report.md                  # narrative report (stages, discoveries, final result)
├── report_analysis.md         # analytical / ablation report (deltas, what helps, what doesn't)
├── run_training.sh            # launch classical sweep (train_all.py) as a background job
├── run_pipeline.sh            # end-to-end pipeline driver
│
├── data/                      # datasets (both raw `text` and `cleaned_text` columns)
│   ├── train_cleaned.csv      # 27,481 tweets (train+val source, stratified 90/10 split at seed 42)
│   ├── test_cleaned.csv       # 3,534 held-out test tweets
│   ├── test.csv               # original unprocessed test file
│   └── output_file.csv        # auxiliary training export
│
├── project/                   # all training / inference / analysis code
│   ├── train_all.py           # classical sweep: 5 model families × N feature configs
│   ├── tune_all.py            # Optuna TPE hyperparameter search over top classical models
│   ├── train_roberta.py       # single-model transformer fine-tuning (any HF checkpoint)
│   ├── ensemble.py            # soft-vote (logit-mean) ensemble over saved transformer checkpoints
│   ├── weighted_ensemble.py   # grid-searched weighted soft-vote variant
│   ├── stacking_ensemble.py   # LR meta-learner stacking variant
│   ├── threshold_tune.py      # per-class bias grid search on ensemble logits
│   ├── classical_dataset.py   # feature builders: TF-IDF, VADER, AFINN, POS tag counts
│   ├── feature_utils.py       # shared feature helpers
│   ├── bert_dataset.py        # transformer Dataset + tokenization
│   ├── sequence_dataset.py    # sequence dataset helpers
│   ├── requirements.txt       # Python dependencies
│   ├── pipeline/              # pipeline glue
│   ├── references/            # reference implementations / notes
│   └── howToUse.txt           # internal run notes
│
├── results/                   # every training / tuning / ensemble run output
│   ├── all_results.csv        # unified 104-row table (classical, transformer, ensemble sections)
│   ├── result_combined.json   # 55-run classical baseline sweep (cleaned text)
│   ├── result_job_raw_*.json  # raw-text classical sweep for raw-vs-cleaned analysis
│   ├── tuned_job_*.json       # Optuna-tuned classical runs
│   ├── roberta_*.json/.log    # individual transformer runs (15 total)
│   ├── ensemble_*.json/.log   # soft-vote ensemble runs (22 total)
│   ├── stacking_*.json        # stacking meta-learner runs
│   ├── weighted_ens_*.json    # weighted-ensemble runs
│   ├── threshold_*.json       # per-class threshold tuning results
│   └── models/                # saved transformer checkpoints (best_model.pt per run)
│
└── logs/                      # stdout/stderr for background jobs launched via run_training.sh
```

## Data

- **`train_cleaned.csv`** has both `text` (raw) and `cleaned_text` columns. All experiments select one via the `--text_col` flag. Transformers consistently prefer `text`; classical char n-grams prefer `cleaned_text`. See `report_analysis.md` §2.3 and §3.2 for the ablation.
- Synthetic metadata columns (`Time of Tweet`, `Age of User`, `Country`, etc.) are uniformly distributed across sentiment classes and carry no signal — excluded from all models.

## Running

**Environment:** conda env `cs4248` (Python 3.x + PyTorch + Transformers + scikit-learn + LightGBM + Optuna). GPU: RTX 5070 Ti used for training.

**Classical sweep:**
```bash
python project/train_all.py \
  --train_csv data/train_cleaned.csv \
  --test_csv  data/test_cleaned.csv \
  --text_col  cleaned_text \
  --job_id    my_run
```

**Single transformer fine-tune:**
```bash
python project/train_roberta.py \
  --model_name vinai/bertweet-base \
  --train_csv  data/train_cleaned.csv \
  --test_csv   data/test_cleaned.csv \
  --text_col   text \
  --job_id     bertweet_01
```

**Ensemble over trained checkpoints:**
```bash
python project/ensemble.py \
  --models "cardiffnlp/twitter-roberta-base-sentiment-latest:results/models/roberta_raw_01/best_model.pt:text,vinai/bertweet-base:results/models/bertweet_01/best_model.pt:text,..." \
  --job_id ens_4_sent_swap
```

## Reports

- [`report.md`](report.md) — full narrative: dataset, feature engineering, model selection, tuning, ensemble experiments, neutral-class analysis.
- [`report_analysis.md`](report_analysis.md) — analytics complement: ablation studies, signed deltas, raw-vs-cleaned verdict per layer, val-set overfitting pattern, and "what to try next".

## Key findings (one-liners)

1. **Character n-grams + VADER** are the biggest classical feature lever (+0.039 F1 over unigram baseline).
2. **Domain-pretrained transformers beat bigger generic ones** — BERTweet-base (135M) > roberta-large (355M) on raw text.
3. **Raw text wins for transformers, cleaned text for char n-grams** — preprocessing is a property of the representation layer, not a universal good.
4. **Tokenizer diversity drives ensemble gains**, not model count. Uniform averaging beats grid-searched weights because 2.7k val samples cannot support fine-grained optimization without overfitting.
5. **The neutral class F1 ceiling (~0.79) is label-noise-bound** and no modeling choice moved it.
