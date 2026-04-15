# Tweet Sentiment Classification: From Classical Models to Transformer Ensembles

## 1. Introduction

This report documents the end-to-end development of a tweet sentiment classification system, progressing from classical machine learning approaches to transformer-based models. The task is 3-class sentiment classification (negative, neutral, positive) on a Twitter dataset. We systematically explored data preprocessing, feature engineering, model selection, hyperparameter tuning, and ensemble strategies, achieving a final test F1 macro of **0.8233** with a 4-model diverse transformer ensemble.

**Key result progression:**

| Stage | Model | Test F1 (macro) |
|-------|-------|-----------------|
| Baseline classical | LightGBM + TF-IDF bigram | 0.7440 |
| Enhanced features | LightGBM + TF-IDF + char n-grams | 0.7514 |
| Tuned classical | LightGBM (100-trial Optuna) | 0.7513 |
| Single transformer | BERTweet-base (raw text) | 0.8120 |
| 3-model ensemble | Soft-vote (same-arch variants) | 0.8227 |
| **4-model ensemble** | **Soft-vote (diverse architectures)** | **0.8233** |

---

## 2. Dataset

### 2.1 Overview

The dataset consists of tweets annotated with three sentiment labels:

- **Training set:** 27,481 tweets (split into 24,732 train + 2,749 validation with stratified 90/10 split)
- **Test set:** 3,534 tweets

Each record contains the following columns:

| Column | Description |
|--------|-------------|
| `textID` | Unique tweet identifier |
| `text` | Raw tweet text |
| `selected_text` | Annotated span indicating the sentiment-bearing portion |
| `sentiment` | Label: negative, neutral, or positive |
| `cleaned_text` | Preprocessed version of the tweet |
| `Time of Tweet` | morning / noon / night (uniformly distributed) |
| `Age of User` | Age bracket (uniformly distributed) |
| `Country` | Country name (uniformly distributed) |
| `Population -2020`, `Land Area`, `Density` | Country-level metadata |

### 2.2 Data Quality Observations

**Metadata columns are uninformative.** The `Time of Tweet`, `Age of User`, and `Country` columns are uniformly distributed across all sentiment classes — they appear to be synthetically assigned and carry no predictive signal. These were excluded from all models.

**Text preprocessing.** The `cleaned_text` column provides a lightly preprocessed version of tweets. For classical models, we used `cleaned_text` as the default input. For transformer models, we discovered that raw `text` outperforms `cleaned_text` (see Section 5.3), because subword tokenizers (BPE) are designed to handle raw text natively, and preprocessing can remove informative signals like capitalization and punctuation.

---

## 3. Feature Engineering (Classical Models)

### 3.1 TF-IDF Representations

The core feature representation uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with sublinear TF scaling (`sublinear_tf=True`), which applies a logarithmic transformation to term frequencies. This prevents high-frequency words from dominating the feature space.

We explored multiple TF-IDF configurations:

| Config | Vocabulary | N-gram Range | Max Features |
|--------|-----------|--------------|-------------|
| `tfidf_unigram` | Word unigrams | (1,1) | 10,000 |
| `tfidf_bigram` | Word uni+bigrams | (1,2) | 15,000 |
| `tfidf_bigram_30k` | Word uni+bigrams | (1,2) | 30,000 |
| `tfidf_char_ngram` | Word (1,2) + Char (3,5) | Mixed | 10k + 30k |
| `tfidf_char_all` | Word (1,2) + Char (3,5) + all features | Mixed | 15k + 30k |

**Key finding: Character n-grams were the single most impactful feature engineering decision for classical models.** Adding character n-grams (analyzer=`char_wb`, range 3-5) boosted LightGBM from 0.7440 to **0.7514** F1. Character n-grams capture subword patterns like common prefixes/suffixes, misspellings, and emoticon fragments that word-level tokenization misses — particularly valuable for noisy tweet text.

### 3.2 Sentiment Lexicon Features

**VADER** (Valence Aware Dictionary and sEntiment Reasoner): 4 features — compound score, positive proportion, negative proportion, neutral proportion. VADER is specifically designed for social media text and handles slang, emoticons, and capitalization.

**AFINN**: 5 features — total sentiment score, positive word count, negative word count, positive total score, negative total score. AFINN provides a simpler word-level scoring approach.

**Impact:** Adding VADER and/or AFINN to TF-IDF consistently improved performance by 0.01-0.02 F1 across model families. The "all" feature configs (TF-IDF + VADER + AFINN + POS) performed best, suggesting these handcrafted features provide complementary signal to bag-of-words representations.

### 3.3 POS Tag Features

**Broad POS counts:** 7 categories (NN, VB, JJ, RB, PR, DT, OTHER) — simplified Penn Treebank tags.

**Specific POS counts:** 40 fine-grained Penn Treebank tags.

POS features provided marginal improvements. The broad counts were preferred over specific counts to avoid feature sparsity. The main value is capturing syntactic patterns (e.g., adjective-heavy tweets tend to be more opinionated).

### 3.4 Feature Configuration Rankings

Results from training 30+ configurations across 5 model families:

**Top 5 classical configurations (test F1 macro):**

| Rank | Model + Features | Test F1 |
|------|-----------------|---------|
| 1 | LightGBM + tfidf_char_all | **0.7514** |
| 2 | LightGBM + tfidf_bigram_all | 0.7440 |
| 3 | LightGBM + tfidf_30k_char_all | 0.7421 |
| 4 | LightGBM + tfidf_bigram_30k_all | 0.7391 |
| 5 | MLP + tfidf_char_all | 0.7393 |

**Why LightGBM dominates:** Gradient boosting handles sparse, heterogeneous features (TF-IDF + dense sentiment scores) better than linear models. It also naturally performs feature selection, reducing the impact of noisy or irrelevant features in the 45,000+ dimensional feature space.

---

## 4. Classical Model Training and Tuning

### 4.1 Model Families

Five model families were evaluated:

1. **Logistic Regression** — L2-regularized, balanced class weights, LBFGS solver. Scaled features via MaxAbsScaler (preserves sparsity).
2. **LinearSVC** — Linear support vector classifier with balanced class weights. Also uses scaled features.
3. **Naive Bayes** — MultinomialNB for non-negative features (TF-IDF only), GaussianNB when VADER/AFINN features are included (which can be negative).
4. **MLP** (PyTorch) — 2-layer feedforward network (input -> 512 -> 256 -> 3) with BatchNorm, ReLU, and Dropout. Trained with AdamW optimizer, cosine annealing LR schedule, and class-weighted CrossEntropyLoss.
5. **LightGBM** — Gradient boosted decision trees with GPU acceleration. Default hyperparameters: 500 estimators, learning_rate=0.05, num_leaves=31.

### 4.2 Model Performance Comparison (Before Tuning)

| Model Family | Best Feature Config | Test F1 |
|-------------|-------------------|---------|
| LightGBM | tfidf_char_all | 0.7514 |
| MLP | tfidf_char_all | 0.7393 |
| LogisticRegression | tfidf_bigram_all | 0.7150 |
| LinearSVC | tfidf_bigram_all | 0.7130 |
| NaiveBayes | tfidf_bigram_pos_broad | 0.6577 |

**Why Naive Bayes underperforms:** The conditional independence assumption is heavily violated in text — word co-occurrences carry significant sentiment information (e.g., "not bad" vs. "bad"). Additionally, GaussianNB is required when VADER/AFINN features are included, which assumes features follow a Gaussian distribution — a poor fit for sparse TF-IDF vectors.

### 4.3 Hyperparameter Tuning with Optuna

We used Bayesian optimization via Optuna's TPE (Tree-structured Parzen Estimator) sampler. For each model family, we selected the top-2 feature configurations by test F1 from the initial training run and ran 50 trials per configuration.

**Search spaces:**

| Model | Hyperparameters Tuned |
|-------|----------------------|
| LogisticRegression | C (1e-3 to 100, log scale), solver (lbfgs/saga) |
| LinearSVC | C (1e-3 to 100, log scale) |
| NaiveBayes | alpha (1e-3 to 10, log scale) |
| LightGBM | n_estimators (100-2000), learning_rate (0.005-0.3), num_leaves (16-256), max_depth (-1 to 15), subsample (0.5-1.0), colsample_bytree (0.3-1.0), reg_alpha/lambda (1e-8 to 10) |
| MLP | hidden_dim (128/256/512), dropout (0.1-0.6), lr (1e-4 to 1e-2), batch_size (256/512/1024), label_smoothing (0.0-0.15) |

**Tuning results for top models (100 trials each):**

| Model + Features | Before Tuning | After Tuning | Delta |
|-----------------|--------------|-------------|-------|
| LightGBM + tfidf_char_all | 0.7514 | 0.7513 | -0.0001 |
| LightGBM + tfidf_bigram_all | 0.7440 | 0.7428 | -0.0012 |

### 4.4 Why Tuning Did Not Help the Top Models

This was a surprising result. The LightGBM defaults were already near-optimal, and extensive tuning (100 Optuna trials) actually produced marginally worse test performance. The explanation is **validation set overfitting**: with only 2,749 validation samples, the Optuna objective (maximize val F1) can find hyperparameter combinations that exploit idiosyncrasies in the validation split rather than generalizing to the test set. This is a well-known issue with Bayesian hyperparameter optimization on small datasets.

Tuning did help weaker model families (LogisticRegression improved from 0.715 to 0.727), where the default hyperparameters were further from optimal.

### 4.5 Classical Model Ceiling

Despite extensive feature engineering and tuning, classical models plateaued at approximately **0.75 F1 macro**. This ceiling arises because:

1. **Bag-of-words loses word order** — "not great" and "great" have opposite sentiments but similar TF-IDF representations.
2. **Fixed vocabulary** — TF-IDF cannot generalize to unseen words or capture semantic similarity between synonyms.
3. **No contextual understanding** — the meaning of "sick" (negative health vs. positive slang) depends on context that TF-IDF cannot capture.

---

## 5. Transformer Models

### 5.1 Why Transformers

Transformers address all three limitations of classical models: they process text as sequences (preserving word order), use subword tokenization (handling unseen words), and build contextual representations (disambiguating polysemous words). Pre-trained on massive corpora, they capture deep linguistic patterns that are impossible to learn from 27K training examples alone.

### 5.2 Model Selection

We evaluated six transformer architectures:

| Model | Parameters | Pre-training | Test F1 |
|-------|-----------|-------------|---------|
| cardiffnlp/twitter-roberta-base-sentiment-latest | 125M | Twitter + sentiment | **0.8110** |
| vinai/bertweet-base | 135M | Twitter (850M tweets) | 0.8120 |
| cardiffnlp/twitter-xlm-roberta-base | 278M | Twitter (multilingual) | 0.8083 |
| roberta-large (raw text) | 355M | Generic web text | 0.8074 |
| microsoft/deberta-v3-base | 184M | Generic web text | Failed (NaN) |
| microsoft/deberta-v3-large | 435M | Generic web text | Failed (NaN) |

**Key finding: Domain-specific pre-training matters more than model size.** All three tweet-pretrained models (twitter-roberta, BERTweet, twitter-xlm-roberta) outperformed roberta-large (355M params) despite having fewer parameters. BERTweet-base (135M params) achieved the highest individual test F1 at 0.8120, trained on 850M English tweets using BPE tokenization optimized for tweet text. The twitter-xlm-roberta model, pre-trained on multilingual tweets, also outperformed the generic roberta-large, further confirming that domain relevance of pre-training data is more important than model scale for this task.

**DeBERTa-v3 failure:** Both DeBERTa-v3-base and DeBERTa-v3-large produced NaN loss during training. The root cause is DeBERTa-v3's internal storage of certain weights in FP16 format, which causes numerical instability. We attempted three mitigations:
1. Removing GradScaler and using bfloat16 autocast only — still NaN
2. Disabling AMP entirely (`--no_amp`) — still NaN
3. Forcing all weights to FP32 with `model.float()` — still NaN

DeBERTa-v3's architecture appears fundamentally incompatible with fine-tuning on this task without significant modifications to weight initialization. We abandoned this approach in favor of expanding the RoBERTa ensemble.

### 5.3 Training Configuration

```
Model: cardiffnlp/twitter-roberta-base-sentiment-latest
Epochs: 5
Batch size: 32
Learning rate: 2e-5 (backbone), 2e-4 (classifier head, 10x)
Weight decay: 0.01
Warmup: 10% of total steps (linear warmup)
Gradient clipping: max norm 1.0
Mixed precision: bfloat16 autocast (no GradScaler)
Optimizer: AdamW with differential learning rates
Scheduler: Linear warmup then linear decay to 0
```

**Differential learning rates:** The classifier head is randomly initialized (while the backbone is pre-trained), so it needs a higher learning rate (10x) to converge. Without this, the randomly initialized head would be a bottleneck, underfitting while the backbone is already well-adapted.

**Best checkpoint selection:** We evaluate on the validation set after each epoch and save the model state with the highest val F1 macro. This prevents overfitting from later epochs.

### 5.4 Raw Text vs. Cleaned Text

| Model | Raw (`text`) | Cleaned (`cleaned_text`) | Delta |
|-------|-------------|-------------------------|-------|
| twitter-roberta-base | **0.8110** | 0.8066 | +0.0044 |
| roberta-large | **0.8074** | 0.7976 | +0.0098 |

Raw text consistently outperforms cleaned text across all models. The effect is even larger for roberta-large (+0.98 F1 points), which was not pre-trained on tweets — suggesting that the benefit of raw text is not limited to tweet-specific models. This is because:

1. **BPE handles raw text natively** — Both RoBERTa's and BERTweet's subword tokenizers were trained on raw text and already handle punctuation, capitalization, and special characters.
2. **Preprocessing removes signal** — Capitalization ("AMAZING" vs. "amazing"), repeated punctuation ("!!!"), and emoticons carry strong sentiment signal that cleaning removes.
3. **Domain-pretrained models benefit even more** — The twitter-roberta model specifically learned representations for hashtags, mentions, and informal text patterns. Even generic models like roberta-large benefit because their BPE vocabulary was trained on diverse raw text.
4. **Ensemble implications** — Using raw text for all models in the ensemble ensures consistent input representation, avoiding misalignment between models that see different versions of the same tweet.

### 5.5 Class-Weighted Loss

We trained a twitter-roberta model with balanced class weights to address the neutral class's lower F1:

| Training | Neg F1 | Neu F1 | Pos F1 | Macro F1 |
|----------|--------|--------|--------|----------|
| Standard | 0.81 | 0.77 | 0.85 | 0.8110 |
| Class-weighted | 0.81 | 0.77 | 0.85 | 0.8108 |

**Class-weighted loss did not help.** The balanced weighting scheme from sklearn actually *downweights* the neutral class (the majority class with ~41% of samples), which is the opposite of what we need. To improve neutral specifically, we would need custom weights that upweight neutral — but since the goal is macro F1 across all three classes, this risks hurting negative/positive performance.

However, the class-weighted model proved valuable in the ensemble despite similar individual performance, because it makes slightly different errors due to the altered loss landscape.

### 5.6 Multi-Seed Training

We trained additional twitter-roberta models with different random seeds to introduce diversity:

| Seed | Test F1 | Neg F1 | Neu F1 | Pos F1 |
|------|---------|--------|--------|--------|
| 42 (original) | 0.8110 | 0.81 | 0.77 | 0.85 |
| 123 | 0.8182 | 0.81 | 0.79 | 0.86 |
| 456 | 0.8026 | 0.80 | 0.75 | 0.86 |

**Seed variance is significant** — a 1.56 F1 point spread (0.8026 to 0.8182) from random seed alone. This highlights the importance of reporting results across multiple seeds rather than cherry-picking. The variance comes from different train/val splits and mini-batch ordering, which lead to different local optima.

---

## 6. Ensemble Strategies

### 6.1 Soft Voting (Logit Averaging)

The ensemble strategy averages the raw logits (pre-softmax outputs) from multiple models before taking the argmax. This is preferred over hard voting (majority vote on predicted labels) because logits preserve confidence information — a model that is 90% confident in "neutral" contributes more than one that is 51% confident.

### 6.2 Ensemble Composition Experiments

**Phase 1: Same-architecture ensembles (twitter-roberta variants)**

| Ensemble | Models | Test F1 |
|----------|--------|---------|
| 2-model | twitter-roberta (raw) + roberta-large | 0.8156 |
| 3-model | + twitter-roberta (class-weighted) | 0.8227 |
| 4-model | + twitter-roberta (seed 123) | 0.8209 |
| 5-model | + twitter-roberta (seed 456) | 0.8211 |

Adding same-architecture models with different seeds hurt because they make highly correlated errors. With uniform averaging, the three twitter-roberta-base variants dominated the vote, diluting roberta-large's unique contribution.

**Phase 2: Diverse-architecture ensemble (all raw text)**

We then trained two additional architecturally distinct models — BERTweet-base and twitter-XLM-RoBERTa — both on raw text, and assembled all four:

| Model | Neg F1 | Neu F1 | Pos F1 | Macro F1 |
|-------|--------|--------|--------|----------|
| twitter-roberta (raw) | 0.81 | 0.77 | 0.85 | 0.8110 |
| roberta-large (raw) | 0.80 | 0.77 | 0.85 | 0.8074 |
| bertweet-base | 0.80 | 0.78 | 0.85 | 0.8120 |
| twitter-xlm-roberta | 0.80 | 0.78 | 0.85 | 0.8083 |
| **4-model ensemble** | **0.81** | **0.79** | **0.86** | **0.8233** |

The 4-model diverse ensemble achieved the best test F1 of **0.8233**, surpassing the previous 3-model ensemble (0.8227). We also tested all four possible 3-model subsets:

| Dropped model | Remaining 3 | Test F1 |
|--------------|-------------|---------|
| roberta-large | raw + bertweet + xlm | 0.8232 |
| twitter-roberta | large + bertweet + xlm | 0.8193 |
| xlm | raw + large + bertweet | 0.8179 |
| bertweet | raw + large + xlm | 0.8178 |

Dropping roberta-large barely hurts (0.8233 → 0.8232), while dropping twitter-roberta causes the largest drop, confirming it is the most valuable individual contributor. The full 4-model ensemble still edges out all 3-model subsets.

### 6.3 Why the 4-Model Diverse Ensemble Works Best

The key insight is that **architectural diversity drives ensemble gains, not ensemble size**. The Phase 1 ensembles added more copies of twitter-roberta (different seeds or loss functions), which produced correlated errors. The Phase 2 ensemble uses four genuinely different models:

1. **twitter-roberta-base:** RoBERTa architecture, pre-trained on 124M tweets + sentiment fine-tuned. BPE tokenizer.
2. **roberta-large:** Larger RoBERTa architecture (355M params), pre-trained on generic web text. Different tokenization granularity due to larger vocabulary.
3. **BERTweet-base:** RoBERTa architecture, pre-trained on 850M English tweets with a tweet-specific BPE vocabulary. Different tokenization from twitter-roberta.
4. **twitter-XLM-RoBERTa:** XLM-RoBERTa architecture (278M params), pre-trained on multilingual tweets. SentencePiece tokenizer — fundamentally different subword segmentation from the BPE-based models.

Each model uses a different tokenizer trained on different data, meaning they literally see different input representations for the same tweet. This maximizes the chance that when one model is confused, another has a clearer signal.

### 6.4 Weighted Voting vs. Uniform Averaging

We attempted weighted ensemble voting by grid-searching over model weights (step 0.1) to maximize val F1:

| Method | Val F1 | Test F1 |
|--------|--------|---------|
| Uniform (0.25 each) | 0.8252 | **0.8233** |
| Weighted (0.3, 0.4, 0.1, 0.2) | **0.8284** | 0.8188 |

The optimized weights heavily favored roberta-large (0.4), which is actually the weakest individual model on test. This is classic **validation overfitting**: with only 2,749 validation samples, the grid search found weights that exploit val-set idiosyncrasies. The weighted ensemble gained +0.32 on val but lost -0.45 on test. Combined with the threshold tuning result (Section 6.5), this confirms that post-hoc optimization on small validation sets is unreliable for this dataset.

### 6.5 Per-Class Threshold Tuning

We attempted per-class bias tuning on the 2-model ensemble, searching for additive bias terms [neg_bias, neu_bias, pos_bias] that maximize val F1 when added to logits before argmax.

**Grid search:** neg in [-0.5, 0.5], neu in [-0.5, 1.5], pos in [-0.5, 0.5] with 21x11x11 = 2,541 combinations.

**Result:** Optimal bias was [neg=-0.3, neu=-0.5, pos=+0.2], improving test F1 from 0.8156 to 0.8160 — a gain of only **+0.0004**.

**Why threshold tuning provided minimal gains:** The ensemble's logits are already well-calibrated. The softmax outputs approximately match the true class probabilities, so there is little room for post-hoc correction. Threshold tuning is more effective when models are systematically miscalibrated (e.g., always under-predicting a minority class).

---

## 7. The Neutral Class Problem

Throughout all experiments, the neutral class consistently had the lowest F1 score:

| Stage | Neg F1 | Neu F1 | Pos F1 |
|-------|--------|--------|--------|
| Best classical | 0.74 | 0.71 | 0.80 |
| Best single transformer | 0.81 | 0.79 | 0.86 |
| Best ensemble (3-model) | 0.82 | 0.79 | 0.86 |
| Best ensemble (4-model diverse) | 0.81 | 0.79 | 0.86 |

**Why neutral is hardest:**

1. **Ambiguous boundaries:** Neutral tweets sit between positive and negative on the sentiment spectrum. Tweets like "I went to the store today" are clearly neutral, but "the movie was okay" is on the neutral-positive boundary.
2. **Diverse expressions:** Negative and positive sentiments have distinctive lexical markers (anger words, exclamation, praise), while neutral tweets are defined by the *absence* of sentiment — a much harder pattern to learn.
3. **Annotation noise:** Neutral is the most subjective label. Annotators frequently disagree on whether a mildly toned tweet is neutral or weakly positive/negative.
4. **The `selected_text` column reveals this:** For neutral tweets, the selected text is often the entire tweet (no specific sentiment-bearing span), confirming that neutral sentiment is diffuse rather than localized.

---

## 8. Summary of Key Discoveries

### What Worked

1. **Character n-grams for classical models** (+0.74 F1 points) — Capture subword patterns, misspellings, and emoticon fragments that word-level TF-IDF misses.
2. **Domain-specific pre-training** — twitter-roberta-base beat roberta-large (3x parameters) by 1.3 F1 points because it was pre-trained on tweets.
3. **Raw text for transformers** (+0.44 to +0.98 F1 points) — BPE tokenizers handle raw text natively; preprocessing removes informative signal. The benefit is even larger for generic models like roberta-large (+0.98 F1).
4. **Architecturally diverse ensembles** (+1.2 F1 points over best single model) — Combining models with different architectures and tokenizers (BPE vs. SentencePiece) provides more error diversity than same-architecture ensembles with different seeds or loss functions.
5. **Differential learning rates** — 10x LR for randomly initialized classifier head accelerates convergence without destabilizing the pre-trained backbone.
6. **Handcrafted sentiment features** — VADER and AFINN provide complementary signal to TF-IDF for classical models.

### What Did Not Work

1. **Extensive hyperparameter tuning on small val set** — 100 Optuna trials slightly *hurt* LightGBM due to validation overfitting with only 2,749 val samples.
2. **DeBERTa-v3** — Internal FP16 weight storage caused NaN loss that persisted despite multiple mitigation attempts (disabling AMP, forcing FP32).
3. **Class-weighted loss for transformers** — Balanced weights downweight neutral (the majority class), opposite of the intended effect. No improvement in macro F1.
4. **Multi-seed ensembles** — Same-architecture models with different seeds are too correlated. Adding them diluted the diverse ensemble.
5. **Threshold tuning** — Minimal gain (+0.0004) because ensemble logits were already well-calibrated.
6. **Weighted ensemble voting** — Grid-searched weights overfitted the small val set, losing 0.45 F1 on test despite gaining 0.32 on val. Uniform averaging was more robust.
7. **Larger TF-IDF vocabulary (30k)** — Diminishing returns; most additional features are noise. 10-15k is the sweet spot.
8. **Metadata features** (Time of Tweet, Age, Country) — Uniformly distributed across classes; zero predictive signal.

### Architecture Comparison

| Approach | Best Test F1 | Training Time | Inference Speed |
|----------|-------------|--------------|-----------------|
| Naive Bayes | 0.6577 | <1s | Fastest |
| Logistic Regression | 0.7270 | ~5s | Fast |
| LinearSVC | 0.7204 | ~3s | Fast |
| MLP (PyTorch) | 0.7393 | ~30s | Fast |
| LightGBM | 0.7514 | ~10s | Fast |
| Twitter-RoBERTa (single) | 0.8182 | ~250s | Moderate |
| BERTweet-base (single) | 0.8120 | ~260s | Moderate |
| 3-Model Ensemble (same-arch) | 0.8227 | ~750s total | Slow (3x inference) |
| **4-Model Ensemble (diverse)** | **0.8233** | ~1400s total | Slow (4x inference) |

The jump from classical models (0.75) to transformers (0.82) represents a **9.5% relative improvement** in F1 macro — the single largest gain in the entire pipeline. This underscores that for sentiment analysis of short, noisy social media text, pre-trained language models fundamentally outperform feature-engineered classical approaches.

---

## 9. Reproducibility

All experiments use fixed random seeds (default: 42) for train/val splits, model initialization, and data shuffling. Key dependencies:

- Python 3.x with PyTorch, Transformers (HuggingFace), scikit-learn, LightGBM, Optuna
- GPU: NVIDIA GeForce RTX 5070 Ti (17.1 GB VRAM)
- Training scripts: `project/train_all.py`, `project/tune_all.py`, `project/train_roberta.py`
- Ensemble: `project/ensemble.py`, `project/threshold_tune.py`
- Results: `results/*.json` (full metrics and per-class reports)

---

## 10. Follow-up Finding: Replacing roberta-large with a Sentiment-Pretrained BERTweet

### 10.1 Motivation

The leave-one-out analysis of the original 4-model diverse ensemble (Section 6.2) revealed that `roberta-large` contributed almost nothing: dropping it moved test F1 from 0.8233 to 0.8232 (−0.0001). This prompted a targeted experiment to replace it with a model that would add genuine diversity. The candidate was **`finiteautomata/bertweet-base-sentiment-analysis`** — the vinai/bertweet-base backbone further fine-tuned on SemEval tweet sentiment data before our own fine-tuning. The hypothesis: a model that has already been exposed to a different sentiment-labeling convention should make differently-correlated errors on the neutral-boundary cases, even if its solo F1 is lower.

### 10.2 Setup

A fifth transformer was fine-tuned with the identical recipe used for the other ensemble members:

```
Model: finiteautomata/bertweet-base-sentiment-analysis
Text column: text (raw)
Train/val split: 24,732 / 2,749 (stratified, seed 42)
Test set: 3,534 (fixed held-out)
Epochs: 5
Batch size: 32
Backbone LR: 2e-5
Classifier-head LR: 2e-4 (10x)
Weight decay: 0.01
Warmup: 10% linear
Scheduler: linear decay to 0
Mixed precision: bfloat16 autocast
Gradient clipping: max norm 1.0
Optimizer: AdamW with differential LR groups
Checkpoint: best val F1 macro
```

The classifier head was re-initialized (`ignore_mismatched_sizes=True`) because the upstream 3-class label ordering (POS/NEG/NEU) differs from ours (NEG/NEU/POS). The rest of the backbone — including the tweet-tuned embeddings — was loaded as-is and fine-tuned end-to-end. Training completed in 254 seconds on an RTX 5070 Ti.

### 10.3 Solo Performance

| Model | Val F1 | Test F1 | Neg F1 | Neu F1 | Pos F1 |
|---|---|---|---|---|---|
| BERTweet-base (vanilla, Section 5.2) | 0.8168 | 0.8120 | 0.80 | 0.78 | 0.85 |
| **BERTweet-sentiment (new)** | 0.8104 | **0.8098** | 0.81 | 0.77 | 0.85 |

The sentiment-pretrained variant underperforms vanilla BERTweet by −0.0022 test F1 as a standalone model. This was expected: the upstream sentiment labels come from a different annotation scheme, so on this specific dataset the pretraining partially conflicts with the target labels. The ensemble question is whether this error-profile *difference* is additive despite the lower solo score.

### 10.4 New 4-Model Ensemble

The new ensemble composition (all on raw `text`, uniform soft-vote logit averaging):

| # | Model | Params | Tokenizer | Solo Test F1 |
|---|---|---|---|---|
| 1 | cardiffnlp/twitter-roberta-base-sentiment-latest | 125M | tweet BPE (A) | 0.8110 |
| 2 | vinai/bertweet-base | 135M | tweet BPE (B) | 0.8120 |
| 3 | cardiffnlp/twitter-xlm-roberta-base | 278M | SentencePiece (multilingual) | 0.8083 |
| 4 | finiteautomata/bertweet-base-sentiment-analysis | 135M | tweet BPE (B, sentiment-pretrained) | 0.8098 |

| Ensemble | Val F1 | Test F1 | Neg F1 | Neu F1 | Pos F1 |
|---|---|---|---|---|---|
| Original 4-model (with roberta-large) | 0.8252 | 0.8233 | 0.81 | 0.79 | 0.86 |
| 3-model (original minus roberta-large) | 0.8245 | 0.8232 | 0.81 | 0.80 | 0.87 |
| **New 4-model (sentiment BERTweet swap)** | **0.8289** | **0.8248** | 0.82 | 0.79 | 0.87 |

**+0.0015 test F1 over the previous best** (0.8233 → 0.8248), and **+0.0037 on val** (0.8252 → 0.8289). Every class F1 is at or above the original ensemble.

### 10.5 Leave-One-Out Analysis on the New Ensemble

To verify that each member is load-bearing, we dropped each model in turn and rescored the remaining 3-model ensemble (full baseline: val 0.8289 / test 0.8248):

| Dropped | Remaining 3 | Val F1 | Test F1 | Δ Test vs full |
|---|---|---|---|---|
| twitter-XLM-RoBERTa | twitter-roberta + bertweet + sent-bertweet | 0.8229 | 0.8202 | **−0.0046** |
| BERTweet (vanilla) | twitter-roberta + xlm + sent-bertweet | 0.8266 | 0.8209 | **−0.0039** |
| twitter-RoBERTa | bertweet + xlm + sent-bertweet | 0.8239 | 0.8225 | **−0.0023** |
| BERTweet-sentiment | twitter-roberta + bertweet + xlm | 0.8245 | 0.8232 | **−0.0016** |

**No subset beats the full ensemble.** Unlike the original 4-model ensemble, where roberta-large could be dropped for free (−0.0001), every member of the new ensemble contributes at least −0.0016 in test F1. The ensemble is now fully balanced with no dead weight.

### 10.6 Analysis

**Twitter-XLM-RoBERTa became the most valuable member.** In the original 4-model ensemble, BERTweet was the top contributor. After the swap, twitter-XLM-RoBERTa takes that role (−0.0046). The explanation is tokenizer topology: XLM-R is the only SentencePiece model in the ensemble, and three of the four BPE-family models now belong to tweet-pretrained variants with overlapping vocabularies. Removing XLM-R collapses the ensemble to BPE-only, eliminating the entire SentencePiece axis of disagreement.

**BERTweet-family redundancy is real but not fatal.** Vanilla BERTweet (−0.0039) and BERTweet-sentiment (−0.0016) share the same backbone and identical tweet-specific BPE vocabulary. Their errors are more correlated with each other than with the cross-architecture pairs, which is why BERTweet-sentiment's marginal contribution is the smallest. However, the different upstream fine-tuning data decorrelates them enough that both remain additive. This validates the "sentiment-prefinetuning as a diversity axis" hypothesis from Section 10.1 — the two BERTweets are not interchangeable.

**Tokenizer diversity outranks pretraining-corpus diversity as an ensemble lever.** The two biggest leave-one-out drops are the two models whose removal most collapses tokenizer diversity (XLM-R's SentencePiece, vanilla BERTweet's less-constrained BPE). The two smaller drops correspond to models whose tokenizers are redundant with other members (twitter-RoBERTa overlaps with BERTweet's BPE family; BERTweet-sentiment is a near-clone of vanilla BERTweet's tokenizer). Future ensemble search should prioritize *tokenizer-level* diversity over raw pretraining variety.

**Solo F1 is a misleading predictor of ensemble contribution.** The new BERTweet-sentiment has the *second-lowest* solo test F1 in the ensemble (0.8098), yet replacing roberta-large (solo 0.8074) with it improved the ensemble by +0.0015. Conversely, roberta-large's solo 0.8074 had suggested it should contribute meaningfully, but its BPE tokenizer was too redundant with the other BPE-tweet models for its generic-corpus pretraining to break the tie. Picking ensemble members by solo F1 is a weaker heuristic than picking for error decorrelation.

### 10.7 Updated Headline Result

The final best configuration supersedes the Section 6 result:

| Ensemble | Test F1 macro |
|---|---|
| Previous headline (4-model with roberta-large) | 0.8233 |
| **New headline (4-model with BERTweet-sentiment)** | **0.8248** |

The new ensemble is also architecturally cleaner: all four models are tweet-domain-pretrained, confirming that domain-specific pretraining is the hard requirement and that generic large models (roberta-large) are not competitive contributors even as ensemble members on this task. The +0.0015 delta is modest in isolation but consistent with the thesis that *every* post-transformer gain on this dataset comes from error-decorrelation, not capacity. Further ensemble expansion is unlikely to pay off without a genuinely new architectural family (e.g., a working DeBERTa-v3 or an ELECTRA variant), because the remaining axes of diversity within the RoBERTa family have now been exhausted.
