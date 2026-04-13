## 4. Experiments

### 4.1 Experimental Setup

We evaluate sentiment classification on financial tweets (23,358 train / 4,122 val / 3,534 test) using a fixed test split for consistent comparison across all runs (88 completed configurations across 3 sweep batches).

**Pipeline.** Our three-stage pipeline first extracts contextual embeddings, then trains either a linear probe or a fine-tuned transformer classifier, and finally generates evaluation reports. This modular design enables controlled ablations by varying one stage while holding others fixed.

**Models.** We evaluate 7 pretrained encoders spanning general-purpose (BERT, DistilBERT, RoBERTa), domain-adapted (FinBERT for finance, BERTweet for social media), and generative architectures (TinyLlama-1.1B, Qwen2.5-0.5B).

**Hyperparameters.** Default transformer runs use an MLP classification head, cross-entropy loss, encoder LR 2e-5, classifier LR 1e-4, weight decay 1e-2, warmup ratio 0.1, dropout 0.1, and early stopping with patience 2 over 5 epochs max. We sweep unfreezing depth over {0, 2, 4, 12} layers and ablate head type, loss function, learning rate, pooling strategy, metadata features, and sentiment lexicon features.

**Reproducibility.** Feature ablation configs (VADER/AFINN/cleaned) were run 3 times each across separate sweep batches. We report averages where multiple runs exist and note the variance.

---

### 4.2 Results

**Table 1: Encoder Comparison — Linear Probe vs. Fine-tuned Transformer**

| Model | Linear Probe F1 | Best Transformer F1 | Layers Unfrozen | Delta |
|---|---|---|---|---|
| FinBERT | 0.7022 | 0.7885 | 12 | +0.0863 |
| TinyLlama | 0.6675 | 0.7999 | 12 | +0.1324 |
| DistilBERT | 0.6670 | 0.7839 | 4 | +0.1169 |
| BERT | 0.6586 | 0.7902 | 12 | +0.1316 |
| Qwen2 | 0.6373 | 0.7844 | 12 | +0.1471 |
| BERTweet | 0.5947 | 0.8008 | 12 | +0.2061 |
| RoBERTa | 0.5578 | 0.7813 | 4 | +0.2235 |

Linear probes show that **FinBERT's frozen representations are already the most discriminative** (0.702), consistent with its finance-domain pretraining. However, the ranking inverts after fine-tuning: BERTweet achieves the highest fine-tuned F1 (0.801) despite the worst linear probe among BERT-family models (0.595). This +0.206 gap — the largest in the table — suggests BERTweet's tweet-domain representations contain rich task-relevant structure that is poorly accessible to a linear classifier but becomes highly effective once the encoder is allowed to adapt.

---

### 4.3 Ablation Studies

#### 4.3.1 Layer Unfreezing

**Table 2: Effect of Unfreezing Depth** (MLP head, CE loss, std LR, auto pooling, no auxiliary features)

| Model | L=0 | L=2 | L=4 | L=12 | Delta(0 to 12) |
|---|---|---|---|---|---|
| BERTweet | 0.6361 | 0.7748 | 0.7833 | 0.8008 | **+0.1647** |
| Qwen2 | 0.6504 | 0.7286 | — | 0.7844 | +0.1340 |
| BERT | 0.6629 | 0.7682 | 0.7725 | 0.7902 | +0.1273 |
| TinyLlama | 0.6828 | 0.7211 | — | 0.7999 | +0.1171 |
| FinBERT | 0.6960 | 0.7588 | 0.7745 | 0.7885 | +0.0925 |

(Qwen2 and TinyLlama L=4 runs did not complete due to resource limits.)

Two findings stand out. First, **the L=0 to L=2 jump accounts for the majority of gains** (e.g., BERTweet: +0.139 out of +0.165 total), indicating that even minimal adaptation of the top layers is critical. Second, **models with the worst frozen representations benefit most from full unfreezing** — BERTweet gains +0.165 vs. FinBERT's +0.093, converging the final scores to a narrow 0.79–0.80 band. This suggests that given sufficient fine-tuning depth, domain match of pretraining matters less than the model's capacity to reorganize its representations.

#### 4.3.2 Classification Head and Loss Function

**Table 3: Head & Loss Ablation** (L=4, CE loss unless noted, std LR, auto pooling)

| Model | MLP + CE | Linear + CE | MLP + Focal (gamma=1.5) |
|---|---|---|---|
| BERT | 0.7725 | **0.7873** | **0.7930** |
| BERTweet | 0.7833 | **0.7844** | 0.7872 |
| FinBERT | 0.7745 | 0.7704 | 0.7686 |
| Qwen2 | — | 0.7381 | 0.7470 |
| TinyLlama | — | 0.7669 | 0.7591 |

Focal loss provides a notable edge for BERT (+0.021 over MLP+CE) but not for FinBERT (-0.006). The linear head actually outperforms MLP+CE for BERT and BERTweet at L=4, suggesting that the MLP's additional parameters may overfit with limited unfreezing depth. The effect is model-dependent: FinBERT consistently prefers MLP over linear, likely because its domain-adapted representations need the extra nonlinearity for task adaptation.

#### 4.3.3 Learning Rate

**Table 4: Learning Rate Ablation** (L=4, MLP head, CE loss, auto pooling)

| Model | Standard (2e-5 / 1e-4) | Low (1e-5 / 5e-5) | Delta |
|---|---|---|---|
| BERT | 0.7725 | 0.7795 | +0.0070 |
| BERTweet | 0.7833 | 0.7883 | +0.0050 |
| FinBERT | 0.7745 | 0.7650 | -0.0095 |
| Qwen2 | — | 0.7086 | — |
| TinyLlama | — | 0.7212 | — |

Lower learning rates give a small improvement for general-purpose encoders (BERT +0.007, BERTweet +0.005), likely by avoiding overshooting when adapting only 4 layers. However, FinBERT performs worse with low LR (-0.010), suggesting its already-aligned representations need a stronger update signal to break out of local optima for this task. For the decoder-only models (Qwen2, TinyLlama), low LR substantially underperforms their L=2 standard-LR results, indicating these architectures require more aggressive updates at shallow unfreezing depths.

#### 4.3.4 Pooling Strategy

**Table 5: Pooling Ablation** (L=4, MLP head, CE loss, std LR)

| Model | Auto ([CLS]) | Mean Pooling | Delta |
|---|---|---|---|
| BERT | 0.7725 | **0.7862** | +0.0137 |
| BERTweet | **0.7833** | 0.7823 | -0.0010 |
| FinBERT | **0.7745** | 0.7696 | -0.0049 |
| Qwen2 | — | 0.7559 | — |
| TinyLlama | — | 0.7619 | — |

Mean pooling helps BERT (+0.014) but slightly hurts BERTweet and FinBERT. This aligns with how each model was pretrained: BERT's [CLS] token was trained with next-sentence prediction (a relatively weak objective), so averaging all token representations captures more information. BERTweet and FinBERT, being fine-tuned on domain-specific data, have [CLS] tokens that already encode useful global information, making mean pooling redundant or mildly harmful.

#### 4.3.5 Metadata Features

**Table 6: Metadata Feature Ablation** (L=4, MLP head, CE loss, std LR, auto pooling)

| Features | BERT | FinBERT |
|---|---|---|
| Baseline (text only) | 0.7725 | 0.7745 |
| +time_of_tweet | 0.7825 (+0.010) | 0.7772 (+0.003) |
| +age_of_user | 0.7845 (+0.012) | 0.7711 (-0.003) |
| +country | 0.7852 (+0.013) | 0.7666 (-0.008) |
| +all three | **0.7853** (+0.013) | **0.7808** (+0.006) |
| BERTweet +all | **0.7940** (+0.011) | — |

Metadata features provide a consistent +1.0–1.3 F1 point improvement for BERT, with country being the most individually helpful feature (+0.013). For FinBERT, individual metadata features have mixed effects — age and country actually hurt performance — but combining all three yields a modest +0.006 gain. This suggests that FinBERT's domain-specific representations already capture some of the signal that metadata provides for general-purpose encoders.

With low LR, metadata gains are reduced: BERT +all goes from 0.7853 (std) to 0.7800 (low), and FinBERT +all from 0.7808 to 0.7696, indicating that the higher learning rate is needed to effectively integrate the additional input dimensions.

#### 4.3.6 Sentiment Lexicon Features

**Table 7: Sentiment Feature Ablation — Fine-tuned Models** (L=12, MLP head, CE loss; averaged over 3 runs)

| Features | BERTweet | FinBERT | TinyLlama |
|---|---|---|---|
| Baseline (none) | 0.8008 | 0.7885 | 0.7999 |
| +VADER | 0.8054 | 0.7888 | 0.7988 |
| +AFINN | **0.8104** | 0.7971 | 0.7987 |
| +VADER+AFINN | 0.8048 | 0.7935 | 0.7999 |
| +VADER (cleaned) | 0.8066 | 0.7987 | 0.7981 |
| +AFINN (cleaned) | 0.8075 | 0.7970 | 0.7988 |
| +VADER+AFINN (cleaned) | 0.8069 | 0.7943 | 0.7972 |

AFINN provides the most consistent improvement across models (+0.5–1.0 F1 points on average). The best feature combination is model-dependent: BERTweet peaks with AFINN alone on raw data (+0.010 avg), while FinBERT benefits most from VADER on cleaned data (+0.010 avg). Stacking both VADER and AFINN together generally underperforms individual features, likely due to feature redundancy.

The cleaned dataset shows no consistent advantage over raw data for fine-tuned models — gains are within run-to-run variance (~0.005 F1).

**Table 8: Sentiment Feature Ablation — Linear Probes** (FinBERT, averaged over 3 runs)

| Features | FinBERT Linear F1 | Delta vs. baseline |
|---|---|---|
| Baseline | 0.7022 | — |
| +VADER (raw) | 0.7336 | +0.031 |
| +AFINN (raw) | 0.7234 | +0.021 |
| +VADER+AFINN (raw) | **0.7362** | **+0.034** |
| +VADER (cleaned) | 0.7326 | +0.030 |
| +AFINN (cleaned) | 0.7225 | +0.020 |
| +VADER+AFINN (cleaned) | 0.7358 | +0.034 |

For linear probes, the impact of lexicon features is much larger (+2.0–3.4 F1 points vs. +0.5–1.0 for fine-tuned). VADER contributes more than AFINN in the linear probe setting, and combining both yields the best result (0.736). This confirms that **lexicon features complement frozen embeddings more than adapted ones** — fine-tuning already captures much of the signal that lexicon scores provide.

**Table 9: BERT L=4 Focal + Features** (averaged over 3 runs)

| Features | BERT F1 |
|---|---|
| Baseline (no features) | 0.7930 |
| +VADER (raw) | 0.7843 |
| +AFINN (raw) | 0.7829 |
| +VADER+AFINN (raw) | 0.7859 |
| +AFINN (cleaned) | **0.7881** |
| +VADER+AFINN (cleaned) | 0.7840 |

Interestingly, adding sentiment features to BERT L4 with focal loss consistently hurts performance versus the no-feature baseline (0.793). This suggests that focal loss already addresses the class-imbalance signal that lexicon features provide, creating redundancy. The combination of focal loss and lexicon features may over-correct for minority classes.

---

### 4.4 Analysis

**Generalization.** Val–test gaps across completed runs are tight (typically within +/-0.01), with no systematic direction. This confirms that our fixed test split and early stopping procedure yield stable estimates without overfitting to the validation set.

**Run-to-run variance.** For the 30 feature-ablation configs run 3 times each, the standard deviation of test F1 is typically 0.003–0.006, meaning differences smaller than ~0.01 should not be over-interpreted.

**Early stopping.** Best epochs distribute across 1–5 (mode at epoch 2–3), validating our choice of patience=2. Models converge quickly, and extended training risks degradation.

**Ranking of design choices by impact:**

1. **Encoder unfreezing** (+0.10–0.16 F1): By far the most impactful factor. Even unfreezing just 2 layers captures 70–85% of the full-unfreeze gain.
2. **Model choice** (up to +0.02 F1 after fine-tuning): Domain-matched pretraining (BERTweet for tweets, FinBERT for finance) helps, but the gap narrows substantially with full unfreezing.
3. **Metadata features** (+0.006–0.013 F1): Consistent small improvement for BERT; mixed for domain-specific models.
4. **Sentiment lexicon features** (+0.005–0.010 F1 fine-tuned; +0.02–0.03 linear probe): AFINN slightly preferred; stacking hurts.
5. **Loss function** (+/- 0.02 F1): Focal loss helps BERT but is model-dependent; not universally better.
6. **Pooling strategy** (+/- 0.014 F1): Model-dependent; mean pooling helps BERT, hurts domain models.
7. **Learning rate** (+/- 0.010 F1): Lower LR slightly helps general-purpose encoders at shallow depth, hurts domain models.

**Best configuration.** BERTweet + full unfreeze (L=12) + MLP head + CE loss + AFINN features achieves **0.8157 macro-F1** (best single run) / **0.8104** (3-run average), our best result.
