# Adapting Vision-Language Debiasing to NLP: Why CBDC Fails in BERT Text Space

**Group XX | Mentored by YY**

---

## Abstract

We adapt Clean Bias Direction Construction (CBDC), a vision-language debiasing method, to 3-class tweet sentiment analysis on the TSAD dataset (27,480 samples) using BERT-base. CBDC uses adversarial PGD to discover confound directions in embedding space, assuming structurally separable subspaces between class-relevant and spurious attributes — a property that holds in CLIP vision but, as we show, fails in BERT text. We find that 3 of 4 PGD-discovered anchor directions have >0.94 cosine similarity with sentiment prototypes, indicating severe topic-sentiment entanglement. The closed-form debias_vl projection (Chuang et al., 2023) improves macro F1 by +1.3 points over baseline (0.649 vs 0.636), but CBDC's iterative refinement yields no further gain. We propose a sentiment-orthogonal PGD gradient constraint that successfully produces sentiment-neutral confound directions (pos-neg gap reduced 3.4x, from 0.0074 to 0.0022), yet classification remains unchanged — revealing that the bottleneck is anchor construction, not gradient drift. Our analysis provides practical insights into when vision-language debiasing methods transfer to text-only settings and when they do not.

---

## 1. Introduction

Sentiment classifiers trained on social media text can exploit spurious correlations — topic mentions, writing register, or domain-specific vocabulary — rather than genuine sentiment signal. A model might associate cryptocurrency terminology with positive sentiment because bullish tweets dominate training, rather than learning the underlying sentiment expressed.

Recent work in vision-language debiasing has produced methods that discover and remove such confound directions from learned representations. CBDC (Clean Bias Direction Construction, CVPR 2026) uses projected gradient descent (PGD) to adversarially search for bias directions in CLIP's embedding space, then trains the encoder to produce representations orthogonal to them. The method succeeds in vision tasks (Waterbirds, CelebA), where spurious attributes (background texture, hair color) occupy structurally separable subspaces from the target class.

**Research question:** Does the structural separability assumption underlying CBDC transfer from CLIP vision to BERT text embeddings?

We adapt CBDC to NLP sentiment analysis using BERT-base on the TSAD tweet dataset. Our contributions:

1. **A faithful port of CBDC to NLP**, mapping CLIP RN50's single-attention-layer tail (attnpool + c_proj) to BERT's layer 11, preserving the same PGD, loss, and training formulations.
2. **A diagnostic analysis** showing CBDC fails because topic and sentiment are entangled in BERT (cosine similarity up to 0.954 between PGD-discovered directions and sentiment prototypes), violating the separability assumption.
3. **A sentiment-orthogonal PGD gradient constraint** that projects gradients perpendicular to sentiment prototypes. This succeeds mechanically (directions become sentiment-neutral) but does not improve classification, narrowing the failure to anchor construction rather than gradient drift.

We find that closed-form debias_vl projection consistently improves F1 by +1.3 points, while CBDC provides no additional benefit. This informs NLP practitioners about the limits of adapting vision-language debiasing to text.

---

## 2. Related Work

### Debiasing word and sentence embeddings

Bolukbasi et al. (2016) identified gender bias directions in Word2Vec via PCA on gendered word pairs and removed them through linear projection. Ravfogel et al. (2020) extended this to contextual embeddings with Iterative Null-space Projection (INLP), which iteratively removes linearly predictable attributes. Dev et al. (2020) introduced orthogonal projection methods for sentence encoders. Our work follows the "identify-then-project" paradigm but uses adversarial search (PGD) rather than supervised linear classifiers to discover the confound subspace.

### Vision-language debiasing

Chuang et al. (2023) proposed debias_vl (Orth-Cali), a closed-form debiasing projection for CLIP representations. Given word pairs defining the spurious attribute, they compute P = P_0 (lambda * M + I)^{-1}, where P_0 is the orthogonal complement of the spurious subspace and M encodes semantic preservation constraints. We use this as our Phase A to discover initial confound directions.

### CBDC

CBDC (CVPR 2026) extends debias_vl with an adversarial refinement stage. Using bipolar PGD with contrastive loss (L_B) and semantic preservation loss (L_s), it discovers cleaner bias directions than closed-form methods. Section 4.2 states that the method assumes "vision-language models encode semantic attributes in structurally separable subspaces." Our work empirically tests this assumption in text-only BERT embeddings and finds it does not hold.

### Adversarial debiasing in NLP

Elazar and Goldberg (2018) used adversarial classifiers to remove demographic attributes from text representations, finding complete removal difficult. Zhang et al. (2018) proposed predictor-adversary architectures. Our approach differs: PGD operates on prompt embeddings rather than training data, making it unsupervised with respect to confound labels.

### Sentiment analysis

Pre-trained models such as BERT (Devlin et al., 2019) achieve strong sentiment performance when fine-tuned. We use BERT-base-uncased as a frozen feature extractor with a linear probe, following the representation-learning paradigm.

---

## 3. Corpus Analysis & Method

### 3.1 Dataset

We use the TSAD (Twitter Sentiment Analysis Dataset) with 27,480 tweets labeled for 3-class sentiment.

| Class | Count | Proportion |
|---|---|---|
| Negative | 7,781 | 28.3% |
| Neutral | 11,117 | 40.5% |
| Positive | 8,582 | 31.2% |

The class distribution is imbalanced, with neutral dominating. We split 70/15/15 into train (19,236), validation (4,122), and test (4,122). The dataset also includes a `selected_text` field identifying the sentiment-bearing span, which we do not use in our pipeline but could inform future error analysis.

<!-- FIGURE 1: Pipeline diagram (5 phases). You should create this in draw.io or TikZ. -->
<!-- **Figure 1**: Pipeline overview: Embed → CBDC Refine → Project → Classify → Evaluate -->

### 3.2 Pipeline Overview

| Phase | Script | Description |
|---|---|---|
| 1 | `data/embed.py` | Encode tweets through frozen BERT → 768-dim CLS vectors |
| 2 | `cbdc/refine.py` | debias_vl map discovery + CBDC text_iccv training |
| 3 | `pipeline/clean.py` | Orthogonal projection to remove confound components |
| 4 | `pipeline/classify.py` | Logistic regression on debiased embeddings |
| 5 | `pipeline/evaluate.py` | Macro F1, per-class metrics, direction interpretability |

The full pipeline runs in ~14 minutes on a single A100 GPU.

### 3.3 Architecture Mapping: CLIP RN50 → BERT

A key design decision is how to map CBDC's architecture from CLIP to BERT. Both have a single-attention-layer "tail" that we can train while freezing the rest:

| Component | CLIP RN50 | BERT-base |
|---|---|---|
| Frozen body | ResNet conv blocks | Layers 0–10 |
| Trainable tail | attnpool (1 MHA) + c_proj | Layer 11 (7.1M params) |
| PGD target | Features before attnpool | Hidden states h_10 |
| Perturbation | Delta on image features | Delta at CLS token position |

Gradients flow through the tail to the PGD perturbation delta. The backbone (110M params) remains frozen throughout.

### 3.4 Phase A: debias_vl Confound Map Discovery

We construct a prompt grid crossing 3 sentiment labels × 32 topics, e.g., *"a negative text about technology"*, *"a positive text about economics"*. We encode all prompts through frozen BERT:

- **Spurious embeddings** (32 topic prompts): define the subspace to remove
- **Candidate embeddings** (96 crossed prompts): for semantic preservation

The debiasing projection:

```
P = P_0 (lambda * M + I)^{-1}
```

where P_0 = I - V(V^T V)^{-1} V^T is the orthogonal complement of the spurious subspace, M is the semantic preservation matrix from word-pair constraints, and lambda = 1000.

Confound directions are extracted via SVD on (I - P), yielding the top-K=4 singular vectors. These become **bias anchors** for PGD.

### 3.5 Phase B: CBDC text_iccv Training

Each of 100 training epochs:

1. **Bipolar PGD** on class-conditioned prompts (e.g., *"a negative text"*):
   - Positive pole: push CLS representation toward bias anchors
   - Negative pole: push toward anti-anchors
   - Inner loop: 20 sign-SGD steps, L_inf clamp at epsilon=1.0, 10 random restarts
   - Loss: L_B * (1 - w_k) - L_s * w_k, with w_k = 0.92
   - L_B = cross-entropy contrastive loss (temperature=100) over anchor pairs
   - L_s = semantic preservation: ||( z_pert - z_orig ) @ keep^T||^2

2. **Direction extraction**: S = z_adv_pos - z_adv_neg (confound directions per sample)

3. **Encoder update**: match_loss = sum_c (S[c::N] @ cls_em[c].T)^2 penalizes directions aligned with class prototypes; ck_loss penalizes anchor-class alignment.

### 3.6 Sentiment-Orthogonal PGD Constraint (Our Modification)

**Motivation:** CBDC Section 4.2 assumes structurally separable subspaces. In CLIP vision, object class and spurious attributes (background, texture) are near-orthogonal, so PGD naturally discovers clean confound directions. CBDC Section 4.4's L_s loss penalizes output deviation but does not constrain gradient direction — PGD can still step into sentiment space if that reduces L_B.

**Method:** Before each PGD sign step, project the gradient orthogonal to the sentiment prototype subspace:

```
g' = g - (g @ S^T) @ S
```

where S is the QR-orthonormalized matrix of sentiment prototype embeddings (3 × 768). This ensures each PGD step moves only in directions uncorrelated with sentiment, analogous to projected gradient descent onto a constrained feasible set (Madry et al., 2018). The prototypes are recomputed each epoch as the encoder evolves.

---

## 4. Experiments

### 4.1 Setup

- **Encoder**: BERT-base-uncased (110M params, frozen; layer 11 unfrozen during CBDC only)
- **Classifier**: Logistic regression (scikit-learn, max_iter=2000)
- **Metric**: Macro F1 (handles class imbalance)
- **Hardware**: NVIDIA A100 80GB PCIe, ~14 min per pipeline run
- **Hyperparameters**: Matched to CBDC RN50 defaults (epsilon=1.0, 20 PGD steps, step_lr=0.0037, keep_weight=0.92, 10 restarts, 100 epochs, AdamW lr=1e-5)

### 4.2 Conditions

| Condition | Description |
|---|---|
| B1 (raw) | Baseline frozen BERT CLS embeddings |
| D1 (debias_vl) | Closed-form word-pair projection only |
| D2 (CBDC) | CBDC encoder training (PGD + tail fine-tuning) |
| D3 (CBDC+proj) | CBDC encoder + residual direction projection |
| D4 (raw+sent-boost) | Baseline + sentiment prototype amplification |
| D5 (CBDC+sent-boost) | CBDC + sentiment prototype amplification |

Each condition except D2/D3 uses the same frozen embeddings with different projection/amplification applied post-hoc. D2/D3 use embeddings re-encoded through the CBDC-trained encoder.

### 4.3 Main Results

**Table 2: Classification results (macro F1). Best in bold.**

| Condition | Val F1 | Test F1 | Delta vs B1 |
|---|---|---|---|
| B1 (raw) | 0.6636 | 0.6363 | — |
| **D1 (debias_vl)** | **0.6694** | **0.6489** | **+1.26** |
| D2 (CBDC, ortho ON) | 0.6607 | 0.6356 | −0.07 |
| D2 (CBDC, ortho OFF) | 0.6626 | 0.6359 | −0.04 |
| D3 (CBDC+proj, ortho ON) | 0.6626 | 0.6331 | −0.32 |
| D3 (CBDC+proj, ortho OFF) | 0.6631 | 0.6353 | −0.10 |
| D4 (raw+sent-boost) | 0.6596 | 0.6348 | −0.15 |
| D5 (CBDC+sent-boost) | 0.6612 | 0.6366 | +0.03 |

debias_vl (D1) is the only condition that consistently improves over baseline. All CBDC variants (D2, D3, D5) show no meaningful improvement regardless of the orthogonal PGD constraint.

### 4.4 Per-Class Analysis

**Table 3: Per-class F1 breakdown for B1 vs D1 (best two conditions).**

| Class | B1 Prec | B1 Rec | B1 F1 | D1 Prec | D1 Rec | D1 F1 | Delta F1 |
|---|---|---|---|---|---|---|---|
| Negative | 0.642 | 0.619 | 0.630 | 0.634 | 0.669 | **0.651** | +2.1 |
| Neutral | 0.584 | 0.651 | 0.616 | 0.608 | 0.619 | 0.614 | −0.2 |
| Positive | 0.707 | 0.625 | 0.663 | 0.709 | 0.657 | **0.682** | +1.9 |

debias_vl improves polar sentiment classes substantially (+2.1 negative, +1.9 positive) while leaving neutral roughly unchanged. This is consistent with topic-sentiment confound removal: topic keywords (e.g., "climate", "crypto") disproportionately co-occur with polar tweets, and removing that confound corrects false polar predictions.

Neutral remains the hardest class across all conditions (F1 ~0.61 vs ~0.65 for polar classes), likely because neutral tweets are more heterogeneous and lack strong lexical markers.

### 4.5 Anchor Entanglement Diagnostic

**Table 4: Cosine similarity between Phase A anchors and sentiment prototypes.**

| Anchor | Pole A topics | Pole B topics | max\|cos(anchor, cls_em)\| |
|---|---|---|---|
| 0 | technology, relationships, environment | economics, climate change, safety | 0.944 |
| 1 | work, economics, politics | gaming, social media, privacy | **0.954** |
| 2 | digital privacy, politics, economics | technology, housing, weather | 0.039 |
| 3 | gaming, relationships, entertainment | climate change, economics, safety | 0.942 |

3 of 4 anchors have >0.94 cosine similarity with sentiment prototypes — they are near-collinear with the class subspace. Only anchor 2 (cosine 0.039) is genuinely sentiment-neutral. This means PGD's L_B loss pushes representations toward anchors that already encode sentiment, making it impossible to discover clean confound directions regardless of gradient constraints.

<!-- FIGURE 2: Bar chart of |cos(anchor, cls_em)| for the 4 anchors, with a horizontal dashed line at 0.2 (desired threshold). This is your most compelling visual. -->

### 4.6 Direction Interpretability

We measure whether PGD-discovered directions are sentiment-neutral by projecting test embeddings onto them and computing the mean projection per class.

**Table 5: Direction quality — lower pos-neg gap means more sentiment-neutral.**

| PGD Constraint | neg mean | neu mean | pos mean | pos-neg gap |
|---|---|---|---|---|
| Orthogonal OFF | 0.0020 | −0.0020 | −0.0054 | 0.0074 |
| Orthogonal ON | −0.0007 | −0.0035 | −0.0029 | **0.0022** |

The orthogonal constraint reduces the pos-neg gap by 3.4×, confirming that it successfully prevents PGD gradients from drifting into sentiment space. However, this does not translate to improved classification — the directions are more sentiment-neutral but not more useful for confound removal.

### 4.7 Training Dynamics

**Table 6: CBDC training loop behavior.**

| Metric | Ortho ON | Ortho OFF |
|---|---|---|
| Best epoch | 1 / 100 | 1 / 100 |
| Best selector F1 | 0.5141 | 0.5141 |
| Epoch 100 selector F1 | 0.5115 | 0.5114 |
| Epoch 100 match_loss | 0.0007 | 0.0013 |
| Epoch 100 ck_loss | 0.0006 | 0.0006 |

In both cases, the selector F1 (nearest-centroid classifier used to track encoder quality) is essentially unchanged from epoch 1 to 100, hovering at ~0.514 — barely above the 0.333 random baseline. The training loop does not improve the encoder. The best checkpoint is always epoch 1, meaning the CBDC iterative refinement provides no benefit over the initial state.

---

## 5. Discussion

### RQ1: Why does debias_vl work but CBDC doesn't?

debias_vl operates in closed form on a prompt-defined subspace. The projection P = P_0(lambda * M + I)^{-1} directly removes the spurious subspace without iterative search, sidestepping the entanglement problem. It does not need to discover clean confound directions — it defines them analytically from word pairs.

CBDC relies on PGD to iteratively discover confound directions, which requires the loss landscape to contain directions that are simultaneously (a) informative about topic confounds and (b) orthogonal to sentiment. In BERT text space, these directions may not exist at sufficient magnitude. The removed projection magnitude for CBDC directions is only 0.034, compared to 0.98 for debias_vl — a 29× difference, indicating CBDC finds directions with negligible impact on the embeddings.

The fundamental issue is architectural. CBDC Section 4.2 assumes structurally separable subspaces, which holds in CLIP's vision encoder: object class (bird species) and spurious attributes (background, watermark) are encoded in geometrically distinct regions because visual features are hierarchically composed. In BERT, topic and sentiment are encoded in overlapping, distributed representations — "crypto" and "positive" share many of the same attention patterns and hidden dimensions.

### RQ2: Does the sentiment-orthogonal constraint help?

The constraint succeeds mechanically: the pos-neg gap drops from 0.0074 to 0.0022 (Table 5), confirming gradients are projected away from sentiment space. But classification is unchanged (Table 2). Two explanations:

**Anchor bottleneck.** The anchors themselves are entangled (3/4 have cosine >0.94). The PGD constraint affects gradient direction during the inner loop, but the L_B loss targets (bias_anchors, anti_anchors) remain sentiment-aligned. PGD is being asked to push representations toward sentiment-correlated anchors while moving only in sentiment-orthogonal directions. This results in near-zero effective perturbation — the constraint and the objective are working against each other.

**Absence of discoverable confound subspace.** Unlike CLIP vision, where background textures occupy a geometrically distinct region from object features, topic and sentiment in BERT text may be distributed across the same dimensions. There may be no low-rank subspace that captures topic confounds independently of sentiment. This would explain why debias_vl works (it removes the topic subspace by construction, accepting some sentiment information loss) while CBDC fails (it searches for a clean separation that doesn't exist).

This finding has practical implications: when adapting vision-language debiasing methods to NLP, practitioners should verify the separability assumption before investing in PGD-based approaches. If the confound and target attributes are entangled, closed-form methods like debias_vl may be both simpler and more effective.

### RQ3: What linguistic properties make neutral tweets hard?

Neutral is consistently the hardest class (F1 ~0.61 vs ~0.65 for polar classes). Examining the per-class confusion:

- **Neutral has high recall (0.65) but low precision (0.58)** in B1, meaning many polar tweets are misclassified as neutral. This suggests the model defaults to neutral when uncertain.
- **debias_vl does not improve neutral** (F1 0.616 → 0.614). Since debias_vl removes topic-correlated variance, and neutral tweets are topic-diverse, the projection has less to correct.
- **Negative and positive benefit most from debiasing** because topic keywords (e.g., "economy", "health") disproportionately co-occur with polar sentiment in social media, creating spurious correlations that the projection removes.

This connects to a broader NLP challenge: neutral sentiment is defined by the *absence* of polar indicators, making it inherently harder to classify with embedding-based methods that look for the *presence* of features.

<!-- TODO: Add 3-5 specific tweet examples where B1 and D1 disagree. -->
<!-- Extract from test set: find cases where baseline predicts wrong, D1 predicts right, and vice versa. -->
<!-- This micro-level analysis is explicitly required by the rubric. -->

---

## 6. Conclusion

We adapted CBDC, a vision-language debiasing method, to NLP sentiment analysis on tweets. Our key finding is that the closed-form debias_vl projection improves macro F1 by +1.3 points, but CBDC's iterative PGD refinement provides no benefit. Through diagnostic analysis, we identified the root cause: BERT text embeddings violate CBDC's structural separability assumption, with 3 of 4 discovered anchor directions having >0.94 cosine similarity with sentiment prototypes.

Our proposed sentiment-orthogonal PGD gradient constraint successfully produces sentiment-neutral directions (3.4× reduction in pos-neg gap) but does not improve classification, narrowing the failure to anchor construction. This result is informative for NLP practitioners: vision-language debiasing methods that rely on adversarial direction discovery may not transfer to text embeddings where target and confound attributes are distributed across overlapping dimensions.

**Limitations.** (1) Single dataset (TSAD); generalizability to other corpora is unknown. (2) Linear probe only — non-linear classifiers might better exploit debiased representations. (3) Static topic prompts; data-driven topic discovery might produce better-separated anchors. (4) We address only gradient drift; the anchor entanglement requires a different fix.

**Future work.** (1) Within-class PCA to discover confound directions using class labels, bypassing the anchor construction bottleneck. (2) Constraining anchor instantiation to enforce sentiment-orthogonality at Phase A, not just Phase B. (3) Multi-dataset evaluation (financial tweets, product reviews) to test domain dependence. (4) Non-linear debiasing (e.g., adversarial networks) that may handle entangled subspaces.

---

## References

- Bolukbasi, T., Chang, K.W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. *NeurIPS*.
- Chuang, C.Y., Mroueh, Y., Greenfeld, D., Torralba, A. (2023). Debiasing Vision-Language Models via Biased Prompts. *arXiv:2302.00070*.
- CBDC (2026). Clean Bias Direction Construction. *CVPR 2026*. <!-- Replace with real citation -->
- Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020). On Measuring and Mitigating Biased Inferences of Word Embeddings. *AAAI*.
- Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
- Elazar, Y., & Goldberg, Y. (2018). Adversarial Removal of Demographic Attributes from Text. *EMNLP*.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR*.
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. *ACL*.
- Zhang, B.H., Lemoine, B., & Mitchell, M. (2018). Mitigating Unwanted Biases with Adversarial Learning. *AIES*.

---

## Appendix A: Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| PGD epsilon (L_inf bound) | 1.0 | CBDC RN50 default |
| PGD steps | 20 | CBDC RN50 default |
| PGD step size | 0.0037 | CBDC RN50 default |
| PGD restarts | 10 | CBDC RN50 default |
| L_s weight (keep_weight) | 0.92 | CBDC RN50 default |
| L_B temperature | 100 | CBDC RN50 default |
| Training epochs | 100 | CBDC RN50 default |
| AdamW learning rate | 1e-5 | CBDC RN50 default |
| debias_vl lambda | 1000 | Chuang et al. 2023 |
| SVD directions (K) | 4 | CBDC RN50 default |
| Classifier | Logistic Regression | sklearn, max_iter=2000 |

## Appendix B: Full Per-Class Reports

### B1 (raw)
```
              precision    recall  f1-score   support
    negative     0.6423    0.6187    0.6303      1167
     neutral     0.5839    0.6511    0.6156      1668
    positive     0.7065    0.6247    0.6631      1287
    macro avg    0.6442    0.6315    0.6363      4122
```

### D1 (debias_vl)
```
              precision    recall  f1-score   support
    negative     0.6344    0.6692    0.6514      1167
     neutral     0.6080    0.6193    0.6136      1668
    positive     0.7089    0.6566    0.6817      1287
    macro avg    0.6504    0.6484    0.6489      4122
```

### D2 (CBDC, ortho ON)
```
              precision    recall  f1-score   support
    negative     0.6445    0.6135    0.6286      1167
     neutral     0.5827    0.6505    0.6147      1668
    positive     0.7032    0.6278    0.6634      1287
    macro avg    0.6435    0.6306    0.6356      4122
```

### D2 (CBDC, ortho OFF)
```
              precision    recall  f1-score   support
    negative     0.6416    0.6153    0.6282      1167
     neutral     0.5859    0.6439    0.6135      1668
    positive     0.6991    0.6356    0.6659      1287
    macro avg    0.6422    0.6316    0.6359      4122
```

## Appendix C: Reproducibility

All experiments can be reproduced from the project repository.

```bash
cd project/

# Full pipeline (debias_vl + CBDC with orthogonal PGD, all evaluation conditions)
python run_all.py

# Resume from specific phase
python run_all.py --start_phase 2

# Skip CBDC, baseline + debias_vl only
python run_all.py --skip_cbdc
```

SLURM submission:
```bash
sbatch submit_new.slurm                                    # full pipeline
sbatch --export=ALL,START_PHASE=2 submit_new.slurm         # resume from phase 2
```

Results are saved to `results/eval_report.txt`. Intermediate artifacts (embeddings, directions, checkpoints) are cached in `cache/{model_slug}/`.

---

<!-- NOTES FOR COMPLETION -->
<!-- 1. ADD Figure 1: Pipeline diagram (draw.io or TikZ) -->
<!-- 2. ADD Figure 2: Bar chart of |cos(anchor, cls_em)| for 4 anchors -->
<!-- 3. ADD 3-5 specific tweet examples for micro error analysis (RQ3) -->
<!-- 4. FILL IN Group number, mentor name, student IDs -->
<!-- 5. ADD Statement of Independent Work (required) -->
<!-- 6. ADD AI Tools disclosure table (required) -->
<!-- 7. CONVERT to LaTeX using the ACL template -->
