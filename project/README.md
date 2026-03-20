# CBDC Financial Sentiment Pipeline

NLP adaptation of **CBDC (Clean Bias Direction Construction)** (CVPR 2026) applied to FinBERT embeddings for tweet financial sentiment classification. The core idea: use an SAE to find a tweet-style direction, then use PGD-based bipolar search to refine it into a clean, semantics-preserving style axis (`delta_star`). Projecting this out of embeddings removes stylistic confounds before classification.

## Quick Start

```bash
cd project/

# Full pipeline (all 7 phases)
python run_all.py

# Or run phases individually:
python data/embed.py          # Phase 1
python sae/sae.py             # Phase 2
python sae/sae_analysis.py    # Phase 3a
python cbdc/refine.py         # Phase 3b
python pipeline/clean.py      # Phase 4
python pipeline/classify.py   # Phase 5
python pipeline/evaluate.py   # Phase 6
```

---

## 7-Phase Pipeline

```
[Tweet corpus]   [Formal corpus]
     │                 │
     └────── Phase 1 ──┘
         data/embed.py
    FinBERT CLS embeddings
    z_tweet_{train,val,test}.pt
    z_formal.pt
         │
     Phase 2
     sae/sae.py
    Sparse Autoencoder (768→1536→768)
    trained on mixed tweet+formal
    sae_model.pt
         │
     Phase 3a
     sae/sae_analysis.py
    style_scores = tweet_acts − formal_acts
    v_style = top-K SAE decoder columns
    v_style.pt, v_shift.pt
         │
     Phase 3b
     cbdc/refine.py  ← CBDC bipolar PGD
    For each anchor batch:
      δ+ → push toward v_style  (tweet pole)
      δ− → push away from v_style (formal pole)
      V_B = normalize(mean(z_pos − z_neg))
    delta_star = normalize(mean(V_B over runs))
    delta_star.pt
         │
     Phase 4
     pipeline/clean.py
    z_clean = normalize(z − (z·d)·d)
    z_tweet_{split}_clean_{direction}.pt
         │
     Phase 5
     pipeline/classify.py
    Linear(768, 3) probe trained per condition
    B1/B2/B2.5/B3/C evaluated, results.pt
         │
     Phase 6
     pipeline/evaluate.py
    Direction interpretability + linearity + report
    results/eval_report_new.txt
```

---

## How SAE and CBDC Work Together

### Step 1 — SAE discovers v_style (Phase 2–3a)

The SAE is trained on a mixture of tweet and formal embeddings. Its decoder matrix `W_d ∈ ℝ^{768×1536}` learns an overcomplete feature dictionary. After training, for each feature `j`:

```
style_score[j] = mean_activation(tweets)[j] − mean_activation(formal)[j]
```

The top-K highest-scoring features are the SAE's best candidates for "tweet style". The style direction is then:

```
v_style = normalize(W_d[:, top_K] @ style_scores[top_K])   # (768,)
```

This gives a *linear* summary of the SAE's learned style features — it points in the embedding direction most associated with tweet-like content.

**Why SAE over plain mean-shift?**
Plain mean-shift (`v_shift = mean(z_tweet) − mean(z_formal)`) mixes style with sentiment signal. The SAE's sparse activations isolate features that are consistently tweet-like *regardless* of sentiment class, giving a cleaner style axis.

### Step 2 — CBDC bipolar PGD refines v_style into delta_star (Phase 3b)

The SAE gives a good but imperfect style direction. CBDC (paper §4.3–4.4) refines it using adversarial search:

**Bipolar PGD** — for each anchor batch from `z_tweet_train`:

| | Positive pole (tweet) | Negative pole (formal) |
|---|---|---|
| **Bias loss L_B** | `-cosine(z+δ⁺, v_style)` | `+cosine(z+δ⁻, v_style)` |
| **Semantic loss L_s** | `‖v_semantic · (z_pert⁺ − z)‖²` | `‖v_semantic · (z_pert⁻ − z)‖²` |
| **Objective** | minimize `L_B + λ_s · L_s` | minimize `L_B + λ_s · L_s` |

Where `v_semantic = normalize(mean(z_formal))` is the formal centroid, serving as the semantic axis to preserve (paper eq. 9).

The **clean direction** per batch (paper eq. 6):
```
V_B = normalize(mean(z_pos − z_neg))
```

Taking the difference cancels noisy concepts shared by both poles, leaving only the style axis. This is CBDC's key insight over monopolar perturbation.

Final aggregation over `n_runs` batches × `n_restarts` random initializations:
```
delta_star = normalize(mean(V_B over all runs))
```

### Step 3 — Orthogonal projection removes style (Phase 4)

```python
z_clean = normalize(z − (z · delta_star) · delta_star)
```

This is paper eq. 2. The style component along `delta_star` is subtracted out, and the result is re-normalized. A linear probe trained on `z_clean` should classify sentiment based on semantics rather than style.

---

## Evaluation Conditions

| Condition | Embedding | Direction removed |
|-----------|-----------|------------------|
| B1 (raw) | Raw FinBERT CLS | — |
| B2 (SAE) | Projected | `v_style` (SAE-derived) |
| B2.5 (mean-shift) | Projected | `v_shift` (plain mean-shift) |
| **B3 (SAE+CBDC)** | Projected | `delta_star` (refined) ← main method |
| C (label-guided) | Projected | `v_label_guided` (oracle, uses labels) |

All evaluated with macro F1 on tweet sentiment (negative/neutral/positive).

---

## Gradient Flow — Technical Detail

This section directly addresses the structural difference between the original CLIP implementation and this NLP adaptation.

### Original CLIP implementation (`simple_pgd.py`)

In the CLIP version, `z = Ψ(x)` is computed *online* — perturbation `δ` is applied to the input (or an intermediate representation), and the gradient `∂L/∂δ` flows **back through the projection layer and the last N transformer resblocks** of the vision encoder.

Computational graph:
```
δ → [last N transformer blocks] → projection layer → z_pert → cosine(z_pert, text_emb) → L
```

The gradient signal carries second-order information about how the encoder maps inputs to the embedding space.

### Our NLP adaptation (this codebase)

We operate on **pre-cached final CLS embeddings**. `z` is the FinBERT CLS token embedding *after all 12 transformer layers have already run*. The cache files (`z_tweet_train.pt`, etc.) are fixed tensors — the FinBERT encoder is not re-invoked during PGD.

The computational graph for `∂L/∂δ` is:

```
δ → z + δ → normalize(z + δ) → cosine(z_pert, v_style) → L
              ↑
        L2-normalization
     (only operation in graph)
```

**There are zero transformer layers in the gradient graph.** The gradient `∂L/∂δ` flows only through the L2-normalization:

```
∂L/∂δ = ∂L/∂z_pert · ∂z_pert/∂(z+δ) · ∂(z+δ)/∂δ
       = ∂L/∂z_pert · J_normalize · I
```

Where `J_normalize = (I − z_pert z_pert^T) / ‖z+δ‖` is the Jacobian of L2-normalization.

### Why is this valid?

Paper §4.2 explicitly states: *"latent-space PGD operates on z = Ψ(x) directly"* — i.e., the perturbation is in the embedding space, not the input space. Our implementation is consistent with this formulation.

The trade-off:
- **CLIP**: richer gradient signal (through encoder layers) → potentially better curvature information for δ optimization
- **Ours**: δ is optimized purely in the linear embedding space → simpler optimization landscape, but also simpler geometry. Since the final objective (linear probe on embeddings) also operates in this same linear space, this is a reasonable compromise.

### Summary table: CLIP vs ours

| | CLIP (original) | Ours (NLP adaptation) |
|---|---|---|
| **z is** | Pre-projection intermediate | Final cached CLS embedding |
| **Gradient flows through** | Projection layer + last N transformer blocks | L2-normalization only |
| **Transformer layers in graph** | N (last few) | 0 |
| **L_B (bias loss)** | Cross-entropy over text prompt pairs | ±cosine(z+δ, v_style) |
| **v_C (semantic axis)** | Neutral concept from text prompts | Formal centroid |
| **Direction per step** | v_style replaces "he"/"she" prompts | v_style from SAE top-K features |
| **Output** | Full subspace S_B | Single delta_star vector |

---

## Compromises vs Original CLIP CBDC

| CLIP version | Our NLP adaptation | Justification |
|---|---|---|
| Online encoder re-pass per PGD step | Pre-cached embeddings, no re-pass | §4.2: latent-space PGD valid; FinBERT re-encoding at every step would be prohibitively slow |
| Contrastive text prompt pairs (e.g. "he"/"she") as L_B target | v_style from SAE decoder as L_B target | No image encoder in NLP; SAE provides the closest analog to CLIP's text prompts |
| Neutral text concept v_C | Formal corpus centroid v_semantic | Formal financial prose is the domain-appropriate "neutral" register |
| Full orthogonal subspace S_B | Single delta_star vector | Sufficient for linear-probe evaluation; single projection is the standard debiasing operation in NLP literature |
| Random restarts = `num_samples` in multi-sample loop | `n_restarts` parameter | Direct correspondence; same purpose |

---

## Configuration

Key hyperparameters (`config.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `epsilon` | 0.10 | PGD perturbation budget (L∞) |
| `n_steps` | 50 | PGD optimization steps per pole |
| `step_lr` | 0.01 | Adam lr for δ |
| `lambda_s` | 0.2 | Semantic preservation weight |
| `n_anchors` | 500 | Anchor batch size per run |
| `n_directions` / `n_runs` | 16 | Independent runs to average |
| `n_restarts` | 3 | Random restarts per run |
| SAE hidden_dim | 1536 | 2× overcomplete dictionary |
| SAE top_k | 32 | Top-K style features for v_style |

---

## File Structure

```
project/
├── config.py              # PGDConfig, SAEConfig, MODEL_REGISTRY
├── encoder.py             # FinBERTEncoder (frozen), encode_with_delta()
├── losses.py              # l_semantic_preservation, l_bias_*
├── dataset.py             # load_tsad(), load_formal_sentences()
├── run_all.py             # Full pipeline orchestrator (7 phases)
├── data/
│   └── embed.py           # Phase 1: encode & cache embeddings
├── sae/
│   ├── sae.py             # Phase 2: train SparseAutoencoder
│   └── sae_analysis.py    # Phase 3a: extract v_style from SAE
├── cbdc/
│   └── refine.py          # Phase 3b: bipolar PGD → delta_star
├── pipeline/
│   ├── clean.py           # Phase 4: orthogonal projection
│   ├── classify.py        # Phase 5: linear probe per condition
│   └── evaluate.py        # Phase 6: full eval report
└── cache/                 # Auto-generated intermediate files
    ├── z_tweet_{split}.pt
    ├── z_formal.pt
    ├── sae_model.pt
    ├── v_style.pt
    ├── v_shift.pt
    ├── delta_star.pt
    └── results.pt
```
