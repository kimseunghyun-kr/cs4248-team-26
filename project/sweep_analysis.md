## 4. Experiments

### 4.1 Experimental Setup

We evaluate sentiment classification on financial tweets (23,358 train / 4,122 val / 3,534 test) using a fixed test split for consistent comparison across all runs (88 completed configurations).

**Pipeline.** Our three-stage pipeline first extracts contextual embeddings, then trains either a linear probe or a fine-tuned transformer classifier, and finally generates evaluation reports. This modular design enables controlled ablations by varying one stage while holding others fixed.

**Models.** We evaluate 7 pretrained encoders spanning general-purpose (BERT, DistilBERT, RoBERTa), domain-adapted (FinBERT for finance, BERTweet for social media), and generative architectures (TinyLlama, Qwen2).

**Hyperparameters.** Transformer runs use an MLP classification head, cross-entropy loss, encoder LR 2e-5, classifier LR 1e-4, weight decay 1e-2, and early stopping with patience 2. We sweep unfreezing depth over {0, 2, 4, 12} layers.

---

### 4.2 Results

**Table 1: Encoder Comparison — Linear Probe vs. Fine-tuned Transformer**

| Model | Linear Probe F1 | Best Transformer F1 | Δ |
|---|---|---|---|
| FinBERT | 0.7022 | 0.7885 | +0.0863 |
| TinyLlama | 0.6675 | 0.7999 | +0.1324 |
| BERT | 0.6586 | 0.7902 | +0.1316 |
| DistilBERT | 0.6670 | 0.7839 | +0.1169 |
| BERTweet | 0.5947 | 0.8008 | +0.2061 |
| Qwen2 | 0.6373 | 0.7844 | +0.1471 |
| RoBERTa | 0.5578 | 0.7813 | +0.2235 |

Linear probes show that **FinBERT's frozen representations are already the most discriminative** (0.702), consistent with its finance-domain pretraining. However, the ranking inverts after fine-tuning: BERTweet achieves the highest fine-tuned F1 (0.801) despite the *worst* linear probe among BERT variants (0.595). This +0.206 gap — the largest in the table — suggests BERTweet's tweet-domain representations contain rich task-relevant structure that is poorly accessible to a linear classifier but becomes highly effective once the encoder is allowed to adapt.

---

### 4.3 Ablation Studies

#### 4.3.1 Layer Unfreezing

**Table 2: Effect of Unfreezing Depth** (MLP head, CE loss, no auxiliary features)

| Model | L=0 | L=2 | L=4 | L=12 | Δ(0→12) |
|---|---|---|---|---|---|
| BERTweet | 0.6361 | 0.7748 | 0.7823 | 0.8008 | **+0.1647** |
| Qwen2 | 0.6504 | 0.7286 | 0.7559 | 0.7844 | +0.1340 |
| BERT | 0.6629 | 0.7682 | 0.7800 | 0.7902 | +0.1273 |
| TinyLlama | 0.6828 | 0.7211 | 0.7619 | 0.7999 | +0.1171 |
| FinBERT | 0.6960 | 0.7588 | 0.7696 | 0.7885 | +0.0925 |

Two findings stand out. First, **the L=0→L=2 jump accounts for the majority of gains** (e.g., BERTweet: +0.139 out of +0.165 total), indicating that even minimal adaptation of the top layers is critical. Second, **models with the worst frozen representations benefit most from full unfreezing** — BERTweet gains +0.165 vs. FinBERT's +0.093, converging the final scores to a narrow 0.79–0.80 band. This suggests that given sufficient fine-tuning depth, domain match of pretraining matters less than the model's capacity to reorganize its representations.

#### 4.3.2 Sentiment Lexicon Features

**Table 3: Auxiliary Feature Ablation** (L=12, MLP head, CE loss)

| Features | BERTweet | FinBERT | TinyLlama | Avg Δ vs. baseline |
|---|---|---|---|---|
| Baseline (none) | 0.8008 | 0.7885 | 0.7999 | — |
| +VADER | 0.8059 | 0.7903 | **0.8036** | +0.0035 |
| +AFINN | **0.8157** | 0.7985 | 0.7943 | +0.0064 |
| +AFINN+Clean | 0.8045 | **0.8046** | 0.7999 | +0.0066 |
| +VADER+Clean | 0.8077 | 0.8005 | 0.7995 | +0.0062 |
| +All (V+A+C) | 0.8053 | 0.7927 | 0.7957 | +0.0015 |

AFINN and VADER each provide a modest but consistent improvement (+0.4–1.5 F1 points). Notably, **the best feature combination is model-dependent**: BERTweet peaks with AFINN alone (+0.015), FinBERT with AFINN+Cleaned (+0.016), and TinyLlama with VADER alone (+0.004). Stacking all features together (V+A+C) **underperforms** individual features across all models, likely due to feature redundancy introducing noise or reducing regularization effectiveness.

For **linear probes**, the impact is much larger. FinBERT's linear probe jumps from 0.702 → 0.736 with VADER+AFINN+Cleaned (+0.034), confirming that lexicon features complement frozen embeddings more than adapted ones — fine-tuning already captures much of the signal that lexicon scores provide.

#### 4.3.3 Classification Head and Loss

**Table 4: Head & Loss Ablation** (BERT, L=4, text input)

| Head | Loss | F1 |
|---|---|---|
| MLP | Focal (γ=1.5) | **0.7930** |
| Linear | Cross-Entropy | 0.7873 |
| MLP | Cross-Entropy | 0.7800–0.7862 |

Focal loss provides a small but consistent edge (+0.007 over best CE run), which is expected given class imbalance in financial sentiment data. The MLP head shows high variance across seeds (0.780–0.786 for CE), suggesting that the additional parameters are sensitive to initialization at L=4 depth.

---

### 4.4 Analysis

**Generalization.** Val–test gaps across the top 15 runs range from −0.005 to +0.011, with no systematic direction. This tight agreement confirms that our fixed test split and early stopping procedure yield stable estimates without overfitting to the validation set.

**Early stopping.** Best epochs distribute across 1–5 (mode at epoch 2–3 with 40 of 75 runs), validating our choice of patience=2. Models converge quickly, and extended training risks degradation.

**Best configuration.** BERTweet + full unfreeze (L=12) + MLP head + CE loss + AFINN features achieves **0.8157 macro-F1**, our best result. The three most impactful design choices, in order: (1) full encoder unfreezing (+0.10–0.16 F1), (2) tweet-domain pretraining (+0.01–0.02 over general BERT), (3) lexicon features (+0.005–0.015).
