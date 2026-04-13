# CBDC Prototype Experiments: Consolidated Results

Source artifacts:
- `candidate/logs/*.log` (19 completed Slurm jobs)
- `candidate/manifest.tsv`
- `results/pca/*.csv`
- `report/codex_journal.md`

Derived local tables:
- `candidate/parsed_job_summary.tsv`
- `candidate/parsed_condition_metrics.tsv`
- `candidate/parsed_class_metrics.tsv`
- `candidate/pca_geometry_summary.tsv`

---

## 4. Experiments

### 4.1 Experimental Setup

We evaluate a CBDC-style representation debiasing pipeline on a three-way tweet sentiment classification task with labels `negative`, `neutral`, and `positive`. The prototype-based experiments use a fixed held-out split from `test.csv` with 23,358 training examples, 4,122 validation examples, and 3,534 test examples. This fixed split is the basis for comparing all conditions (`B1`/`D1`/`D2`/`D2.5`/`D3`/`D4`).

The Slurm logs contain 19 completed jobs in two families:

| Family | Jobs | Split | Purpose |
|---|---:|---|---|
| Supervised linear probe | 537752, 541350 | older random split (4,122 test) | sanity check on condition embeddings via learned linear classifier |
| Prototype CBDC evaluation | remaining 17 jobs | fixed `test.csv` (3,534 test) | prompt-prototype classification under all conditions |

The supervised-probe jobs use a different split and evaluation protocol and should not be merged with the prototype results. The separate `sweep_analysis.md` fine-tuning experiments evaluate supervised transformer fine-tuning (macro-F1 up to ~0.80) and are an independent comparison point.

We evaluate several backbone families:
- **Encoder-family:** `bert-base-cased`, `bert-base-uncased`, `bert-large-cased`, `bert-large-uncased`, `roberta-base`, `roberta-large`, and `bertweet-base`.
- **Decoder-family (exploratory):** `Qwen2.5-3B` and `Gemma4-26B-A4B-it`.

Macro-F1 is the primary metric. Accuracy is reported but treated cautiously because several backbones achieve inflated accuracy by collapsing toward the majority `neutral` class.

### 4.2 Methodology

The goal is to reduce reliance on shortcut or spurious cues while preserving actual sentiment semantics. In this project, "bias" refers not to demographic bias but to dataset-specific sentiment shortcuts: punctuation patterns, text length, emoticons, laughter words, URLs, and shallow topic/style cues that correlate with labels without reflecting the target sentiment concept.

The method adapts the CBDC text-stage debiasing framework (originally designed for CLIP vision-language models) to a text-only NLP setting. The adaptation mirrors the original CLIP text-stage structure closely:
- Frozen backbone body with a trainable final transformer layer as the tail
- Intermediate-latent extraction before the tail
- Bipolar latent PGD to discover bias directions from prompt responses
- Semantic-preservation loss (`L_s`) and cross-knowledge loss (`l_ck`)

The main deviation from the original CLIP path is that the NLP port perturbs a pooled-token latent delta (at the CLS position for encoders, or the last non-pad token for decoders) rather than the full intermediate sequence tensor. For encoder-family models, the sentence embedding is pooled from the first token; for decoder-family models, from the last non-pad token. The decoder experiments should be interpreted as exploratory, since the original CBDC motivation is closer to CLIP-like contrastive representation geometry than to generative language-model geometry.

The prompt bank in `cbdc/prompts.py` defines class prompts (negative/neutral/positive sentiment semantics), keep prompts, target prompts, and style-bias pairs (question marks, exclamation marks, short text, laughter words, emoticons). A key caveat is that some of these "bias" cues also carry genuine sentiment signal, so poor prompt calibration can damage the representation rather than debias it.

We compare the following conditions:

| Condition | Description | Selection / label use |
|---|---|---|
| B1 | Raw frozen backbone embeddings with raw class prompt prototypes. | No method training. |
| D1 | DebiasVL-style closed-form projection from mined topic/style confound directions. | Projection only; no epoch-trained tail. |
| D2 | CBDC prompt-based latent-tail training with fixed prompt banks. | Best checkpoint selected by labeled validation centroid macro-F1. |
| D2.5 | Same CBDC training objective as D2. | Checkpoint selected by lowest prompt loss only (most label-light CBDC variant). |
| D3 | DebiasVL-discovered anchors passed into the CBDC training loop. | DebiasVL discovery plus validation-F1 checkpoint selection. |
| D4 | BiasAdv-inspired adversarial direction discovery followed by CBDC training. | First implementation; not a full BiasAdv replica. |

The debiasing routes differ in how bias directions are obtained:
- **D1** mines topic/style confounds and removes a learned subspace through closed-form projection.
- **D2/D2.5** use a fixed handcrafted CBDC style-bias prompt bank.
- **D3** uses DebiasVL-style discovered anchors, then trains with CBDC.
- **D4** selects hard raw training examples under the prototype classifier, applies latent PGD to find directions that worsen true-label prototype cross-entropy while preserving semantic similarity, compresses the resulting adversarial shifts with SVD, and feeds those discovered directions into the CBDC trainer. D4 uses attacks only to discover anchors; it does not adversarially fine-tune on attacked samples. Current D4 is not full BiasAdv, which would require a separate biased auxiliary predictor and a separate debiased model.

**Supervision profile.** The Phase 2 CBDC training objective is label-free. However, labels enter the pipeline at three points: (1) stratified train/val/test splitting, (2) checkpoint selection in D2 (centroid macro-F1) and topic/anchor mining in D1/D3 (label-entropy filtering), and (3) final evaluation against gold labels. D2.5 removes the label-aware checkpoint selector, making it the cleanest current variant for a label-free-training story. The most accurate framing is: label-free prompt-based training objective with label-aware model selection (D2) or label-free model selection (D2.5), evaluated on labeled sentiment data.

### 4.3 Results

Prototype results are highly backbone-dependent. The cleanest positive CBDC result is on RoBERTa-base, where D2 repeatedly improves over the raw prototype baseline. BERT-family models mostly prefer the simpler D1 projection. Larger encoders and decoder-family models reveal prompt/prototype calibration failures.

**Table 2: Best prototype result per backbone (fixed test split, macro-F1)**

| Model | B1 F1 | Best condition | Best F1 | Note |
|---|---:|---|---:|---|
| roberta-base | 0.3747 | D2 | 0.4760 | Clearest CBDC win |
| bert-base-cased | 0.3785 | D1 | 0.4782 | Projection strongest; D2 also helps |
| bert-base-uncased | 0.3081 | D1 | 0.4087 | Same D1-dominant pattern |
| bert-large-uncased | 0.3654 | D1 | 0.4080 | D1 strongest; D3 competitive |
| bert-large-cased | 0.2781 | D1 | 0.4082 | D1 clearly strongest |
| bertweet-base | 0.3000 | D1 | 0.3441 | Weak prototype geometry despite tweet-domain pretraining |
| roberta-large | 0.2670 | D1 | 0.3048 | D2/D2.5 accuracy spike is pathological neutral collapse |
| Qwen2.5-3B | 0.2058 | D1 | 0.2889 | Raw prototype geometry already neutral-collapsed |
| Gemma4-26B-it | 0.2958 | D1 | 0.3129 | Small decoder-side gain; CBDC transfer weak |

**RoBERTa-base** provides the strongest CBDC evidence across multiple fixed-test runs:

| Condition | Test Acc | Test F1 | Source job |
|---|---:|---:|---|
| B1 | 0.4318 | 0.3747 | 542124/542164/542190/542338 |
| D1 | 0.4709 | 0.4577 | 542124/542164/542190/542338 |
| D2 | 0.4754 | 0.4760 | 542338 (best F1) |
| D2.5 | 0.4601 | 0.4613 | 542124 (best F1) |
| D3 | 0.4369 | 0.3777 | 542338 (best F1) |

D2 macro-F1 ranges from 0.4451 to 0.4760 across runs. D2.5 ranges from 0.4411 to 0.4613. Both consistently exceed the B1 baseline (0.3747), supporting the claim that CBDC-style prompt training improves prototype sentiment geometry on a compatible backbone.

**BERT-base-cased** favors D1, with D2 as a secondary improvement:

| Condition | Test Acc | Test F1 | Source job |
|---|---:|---:|---|
| B1 | 0.4287 | 0.3785 | 542203/568012 |
| D1 | 0.4810 | 0.4782 | 542203/568012 |
| D2 | 0.4689 | 0.4604 | 568012 (best F1) |
| D2.5 | 0.4581 | 0.4074 | 542203 (best F1) |
| D3 | 0.4072 | 0.3578 | 568012 |
| D4 | 0.4298 | 0.3790 | 568012 |

### 4.4 Ablation Studies

#### D2 vs. D2.5: Label-aware vs. label-free checkpoint selection

Both conditions use the same CBDC prompt objective. D2 selects checkpoints by labeled validation centroid macro-F1; D2.5 selects by lowest prompt loss only.

On RoBERTa-base across fixed-test runs:
- D2 reaches up to 0.4760 test macro-F1.
- D2.5 reaches up to 0.4613.
- Both substantially exceed B1 (0.3747).

This ablation supports two conclusions: (1) the prompt-based CBDC objective itself is useful, because D2.5 > B1; (2) labeled checkpoint selection provides an additional gain, because D2 > D2.5.

#### D3: DebiasVL anchors into CBDC

D3 tests whether DebiasVL-style discovered anchors improve the CBDC training stage. Results are mixed and backbone-sensitive:
- On RoBERTa-base, D3 stays near raw (F1 0.3615--0.3777).
- On bert-large-cased, D3 improves over D2 (0.3296 vs. 0.2825) but not over D1 (0.4082).
- On bert-large-uncased, D3 is competitive with D1 (0.3931 vs. 0.4080).

Passing DebiasVL anchors into CBDC is not automatically better than either D1 alone or D2 alone.

#### D4: Adversarial discovery into CBDC

D4 was designed to reduce reliance on handcrafted prompt pairs by discovering adversarial hard-example directions automatically. Across all tested backbones, D4 fails to improve over B1:

| Model | B1 F1 | D4 F1 |
|---|---:|---:|
| bert-base-cased | 0.3785 | 0.3790 |
| qwen25-3b | 0.2058 | 0.1963 |
| gemma4-26b-it | 0.2958 | 0.2613 |
| roberta-large | 0.2670 | 0.2644 |
| bert-large-cased | 0.2781 | 0.2819 |

D4-v1 discovers dominant failure directions rather than useful debiasing anchors. This is a negative result: adversarial discovery followed by CBDC does not yet produce usable debiasing anchors without a separate biased auxiliary predictor and preservation objective.

#### Backbone scaling

Larger backbones do not automatically improve the CBDC/prototype path. The clearest example is roberta-large:

| Condition | Test Acc | Test F1 |
|---|---:|---:|
| B1 | 0.3104 | 0.2670 |
| D1 | 0.3478 | 0.3048 |
| D2 | 0.4072 | 0.2262 |
| D2.5 | 0.4061 | 0.2245 |

The high D2/D2.5 accuracy is caused by neutral collapse: negative recall ~0.011, neutral recall ~0.962, positive recall ~0.044. The method is active but produces badly miscalibrated geometry. This is the strongest cautionary result and demonstrates why accuracy alone is misleading.

#### Decoder-family stress test

Decoder-family experiments use last non-pad token pooling and adapt the decoder tail, but this setup is less natural than CLIP-like contrastive representation training.

Qwen2.5-3B is already collapsed at raw B1 (negative recall 0.0000, neutral recall 0.9832, positive recall 0.0227, F1 0.2058). D1 partially repairs negative recall (to 0.2068, F1 to 0.2889), but D2/D2.5 do not repair the prototype geometry and D4 worsens the collapse (F1 0.1963).

Gemma4-26B-it is less collapsed but still weak (B1 F1 0.2958, D1 F1 0.3129). D2/D2.5 are roughly flat, and D3/D4 are harmful (D3 F1 0.2441, D4 F1 0.2613).

These results probe the boundary of the method rather than validate the main CBDC approach.

### 4.5 Visualization and Geometry Diagnostics

We generated shared-basis PCA and cosine t-SNE plots for the fixed-test prototype conditions across 8 backbones (bert, bert-uncased, bert-large-cased, bert-large-uncased, bertweet, roberta, roberta-large, qwen25-3b). Gemma4-26B-it and D4 are not covered by visualization artifacts.

Each PCA CSV contains 3,534 sample points per condition; each t-SNE CSV contains 1,200 sample points per condition. Both include B1, D1, D2, D2.5, and D3, plus overlay markers for class centroids (X), B1 reference centroids (white circles), and class prompt prototypes (stars).

Interpretation caveats:
- PCA is a shared-basis global movement diagnostic, not the classifier.
- t-SNE is a local-neighborhood diagnostic and should not be treated as original-space geometry.
- Prompt prototypes are embedded jointly in t-SNE plots, so they can affect the layout.

A lightweight projected-geometry diagnostic was computed as the ratio of class-centroid separation to within-class spread:

**Table 3: Mean separation/spread ratio across conditions**

| Model | PCA 2D | PCA 3D | t-SNE 2D | t-SNE 3D |
|---|---:|---:|---:|---:|
| bert | 0.187 | 0.422 | 0.225 | 0.247 |
| bert-large-cased | 0.052 | 0.057 | 0.149 | 0.129 |
| bert-large-uncased | 0.152 | 0.145 | 0.135 | 0.133 |
| bert-uncased | 0.078 | 0.073 | 0.089 | 0.175 |
| bertweet | 0.111 | 0.103 | 0.212 | 0.201 |
| qwen25-3b | 0.109 | 0.112 | 0.135 | 0.169 |
| roberta | 0.155 | 0.143 | 0.119 | 0.152 |
| roberta-large | 0.066 | 0.064 | 0.054 | 0.057 |

The visualizations confirm:
- Representation movement under debiasing is real.
- Class separation remains weak for most models.
- Larger models are not automatically better (roberta-large is especially weak).
- Sentiment remains substantially entangled with style, topic, and backbone-specific factors.

### 4.6 Analysis

The overall finding is not that CBDC universally improves sentiment representations. The method is active but highly sensitive to backbone and prompt calibration.

**RoBERTa-base provides the cleanest positive result.** D2 improves macro-F1 by up to +0.1013 over the raw prototype baseline (0.4760 vs. 0.3747), and D2.5 remains above raw even without labeled checkpoint selection (+0.0866). This is the strongest support for the CBDC-style prompt objective in the prototype evaluation setting.

**BERT-family models mostly favor D1.** The simpler DebiasVL-style projection is currently more stable than learned CBDC tail adaptation for these backbones. D2 still helps on BERT-base (up to 0.4604), but D1 dominates (0.4782).

**Large backbones expose calibration failures.** On roberta-large, D2/D2.5 appear strong by accuracy (~0.407) but collapse toward neutral, producing worse macro-F1 (~0.225) than the raw baseline (0.267). Accuracy alone is misleading; macro-F1 and per-class recall are essential.

**Decoder-family runs are exploratory.** Qwen2.5-3B is already neutral-collapsed before debiasing, while Gemma4-26B-it shows a small D1 gain but weak CBDC transfer. These results are useful for understanding the method's boundary, not for validating the main approach.

**D4-v1 is a negative result.** Simply discovering adversarial hard-example directions does not automatically produce usable debiasing anchors. A more faithful BiasAdv-style version would require a separate biased auxiliary predictor and a preservation objective.

**The main remaining bottleneck is prompt/prototype calibration.** The current prompt bank includes cues that are not purely nuisance features in sentiment data; some are genuine affect carriers (emoticons, laughter, exclamations). This explains why the method helps on some backbones while collapsing others.

### 4.7 Summary

On RoBERTa-base, CBDC-style prompt training gives a repeatable improvement over the raw prototype baseline, with D2 reaching 0.4760 macro-F1 and D2.5 remaining above raw even with label-free checkpoint selection. Across BERT-family and other backbones, D1 is the most reliable intervention, suggesting that the simpler DebiasVL-style projection is currently more stable than the learned CBDC tail on many representations.

However, the method is not automatically beneficial. On roberta-large, D2/D2.5 improve accuracy by collapsing toward neutral, producing worse macro-F1. Decoder-family experiments are exploratory, and D4-v1 is a negative result. Shared PCA and cosine t-SNE confirm real representation movement but weak clean class separation; the sentiment geometry remains substantially entangled with other factors.

The CBDC adaptation is technically working and can improve prototype sentiment geometry, but its success is highly backbone-dependent. The most defensible claim is that the method shows promise on compatible backbones (particularly RoBERTa-base) while exposing prompt/prototype calibration as the primary bottleneck for generalization.
