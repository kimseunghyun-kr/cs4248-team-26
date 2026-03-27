# CBDC Financial Sentiment Pipeline — Program Flow & Call Stack

NLP adaptation of **CBDC (Clean Bias Direction Construction)** (CVPR 2026) applied to FinBERT embeddings for tweet financial sentiment classification.

---

## Top-Level Entry Point

```
run_all.py :: main()
  └─ for each phase: subprocess.run([sys.executable, script_path])
       ├─ Phase 1  data/embed.py
       ├─ Phase 2  sae/sae.py
       ├─ Phase 3a sae/sae_analysis.py
       ├─ Phase 3b cbdc/refine.py
       ├─ Phase 4  pipeline/clean.py
       ├─ Phase 5  pipeline/classify.py
       └─ Phase 6  pipeline/evaluate.py
```

Each phase runs as an independent subprocess. `MODEL_NAME` and `CACHE_DIR` are passed via environment variables so each model (`finbert`, `bert`, `bertweet`) writes to its own cache subdirectory.

---

## Phase 1 — `data/embed.py`

**Purpose:** Encode all corpora with FinBERT, cache embeddings + token tensors to disk.

```
embed.py :: main()
  │
  ├─ FinBERTEncoder.__init__(model_name, device)
  │    ├─ AutoTokenizer.from_pretrained(model_name)
  │    ├─ AutoModel.from_pretrained(model_name)          → self.backbone (12-layer BERT)
  │    └─ for p in backbone.parameters(): p.requires_grad_(False)   ← freeze all
  │
  ├─ load_tsad_dataset() / load_tweet_eval()             → raw tweet text + labels
  ├─ load_formal_sentences()                             → raw formal text
  │
  ├─ for each split (train / val / test):
  │    └─ encode_batch(texts, encoder)
  │         └─ FinBERTEncoder.encode_text(texts)
  │              ├─ tokenizer(texts, padding, truncation) → input_ids, attention_mask
  │              ├─ _get_embeddings(input_ids)
  │              │    └─ backbone.embeddings(input_ids)   → (B, L, 768) word+pos+type embeds
  │              └─ _forward_from_embeds(embeds, mask)
  │                   ├─ backbone(inputs_embeds=embeds, attention_mask=mask)
  │                   │    └─ 12 BertLayer forward passes  → last_hidden_state (B, L, 768)
  │                   └─ F.normalize(last_hidden_state[:, 0, :], dim=-1)  → (B, 768) CLS
  │
  └─ torch.save({embeddings, labels, input_ids, attention_mask}, cache/z_tweet_train.pt)
     torch.save({embeddings, labels, input_ids, attention_mask}, cache/z_tweet_val.pt)
     torch.save({embeddings, labels, input_ids, attention_mask}, cache/z_tweet_test.pt)
     torch.save({embeddings},                                    cache/z_formal.pt)
```

**Outputs:** `z_tweet_{train,val,test}.pt`, `z_formal.pt` — all in 768-dim normalized CLS space.

---

## Phase 2 — `sae/sae.py`

**Purpose:** Train a Sparse Autoencoder on the mixed embedding corpus to learn an overcomplete feature dictionary.

```
sae.py :: main()
  │
  ├─ load z_tweet_train.pt + z_formal.pt  (concatenated → mixed_corpus)
  │
  ├─ SparseAutoencoder.__init__(input_dim=768, hidden_dim=1536)
  │    ├─ encoder: Linear(768, 1536) + ReLU   ← sparse activations
  │    └─ decoder: Linear(1536, 768, bias=False)
  │
  └─ train_loop(model, mixed_corpus)
       for epoch in range(50):
         for batch in DataLoader(mixed_corpus):
           ├─ h = model.encode(z_batch)           → (B, 1536) sparse activations
           ├─ z_hat = model.decode(h)              → (B, 768) reconstruction
           ├─ loss_mse  = MSE(z_hat, z_batch)
           ├─ loss_l1   = lambda_l1 * h.abs().mean()    ← sparsity penalty
           ├─ (loss_mse + loss_l1).backward()
           └─ optimizer.step()
       └─ torch.save(model.state_dict(), cache/sae_checkpoint.pt)
```

**What the SAE learns:** Decoder columns `W_d[:, j] ∈ ℝ^{768}` are 1536 overcomplete feature directions. Sparse activations force the model to use only a few features per embedding, so each feature captures a distinct pattern.

---

## Phase 3a — `sae/sae_analysis.py`

**Purpose:** Use the trained SAE to identify which features correspond to "tweet style" vs "formal style", then construct `v_style`.

```
sae_analysis.py :: main()
  │
  ├─ load sae_checkpoint.pt → SparseAutoencoder
  ├─ load z_tweet_train.pt, z_formal.pt
  │
  ├─ compute_mean_activations(z_tweet, sae)
  │    └─ sae.encode(z_tweet)         → (N_tweet, 1536) activations
  │    └─ mean over N_tweet           → (1536,) mean_tweet_acts
  │
  ├─ compute_mean_activations(z_formal, sae)
  │    └─ sae.encode(z_formal)        → (N_formal, 1536)
  │    └─ mean over N_formal          → (1536,) mean_formal_acts
  │
  ├─ style_scores = mean_tweet_acts - mean_formal_acts   → (1536,)
  │    Each score[j] = "how much more active is feature j on tweets than formal"
  │
  ├─ top_k_indices    = argsort(style_scores, descending=True)[:32]   ← tweet-differential
  ├─ bottom_k_indices = argsort(style_scores, descending=False)[:32]  ← formal-differential
  │
  ├─ v_style = normalize(W_d[:, top_k_indices] @ style_scores[top_k_indices])
  │    └─ weighted sum of top-32 decoder columns → (768,) aggregated tweet-style direction
  │    └─ used ONLY for diagnostics (cosine check) in Phase 3b; NOT the PGD loss target
  │
  ├─ style_anchors = normalize(W_d[:, top_k_indices].T)    → (32, 768)  individual tweet columns
  ├─ anti_anchors  = normalize(W_d[:, bottom_k_indices].T) → (32, 768)  individual formal columns
  │    └─ NLP analog of CLIP's mix_pairs:
  │       instead of a single v_style, the PGD uses K paired anchor directions
  │       → contrastive, multi-anchor, gradient-saturation-aware
  │
  ├─ v_shift = normalize(mean(z_tweet) - mean(z_formal))  → (768,) naive baseline
  │
  └─ torch.save(v_style,       cache/v_style.pt)
     torch.save(v_shift,       cache/v_shift.pt)
     torch.save(style_anchors, cache/style_anchors.pt)
     torch.save(anti_anchors,  cache/anti_style_anchors.pt)
```

**What `v_style` is:** A direction in the final 768-dim CLS embedding space that most separates tweet-like from formal-like content, as learned by the SAE's sparse dictionary. Used only for diagnostics (post-run cosine check) in Phase 3b.

**What `style_anchors` / `anti_anchors` are:** The K individual L2-normalized decoder columns before aggregation. These are the NLP analog of CLIP's `mix_pairs` — a set of distinct directions rather than a single averaged vector. Phase 3b uses these K paired anchors in its cross-entropy L_B, giving a contrastive, multi-anchor, gradient-saturation-aware loss.

---

## Phase 3b — `cbdc/refine.py`  ← CORE

**Purpose:** Refine `v_style` into `delta_star` using bipolar PGD. This is the CBDC algorithm.

```
refine.py :: main()
  │
  ├─ load z_tweet_train.pt      → z_tweet (N, 768), input_ids (N, L), attention_mask (N, L)
  ├─ load z_formal.pt           → z_formal (M, 768)
  ├─ load v_style.pt            → v_style (768,)          ← diagnostics only
  ├─ load style_anchors.pt      → style_anchors (32, 768) ← PGD positive poles
  ├─ load anti_style_anchors.pt → anti_anchors  (32, 768) ← PGD negative poles
  ├─ slice to --n_style_anchors (default 8): style_anchors[:8], anti_anchors[:8]
  ├─ FinBERTEncoder(model_name, device)
  │
  └─ collect_delta_star(z_tweet, input_ids, attention_mask,
                        style_anchors, anti_anchors, v_style, z_formal, cfg, encoder)
       │
       ├─ v_semantic = normalize(z_formal.mean(0))    → (768,)  formal centroid = L_s axis
       │
       └─ for run in range(n_runs=16):
            │
            ├─ indices = randperm(N)[:n_anchors=500]
            ├─ z_batch    = z_tweet[indices]           (500, 768)
            ├─ ids_batch  = input_ids[indices]         (500, L)
            ├─ mask_batch = attention_mask[indices]    (500, L)
            │
            └─ for restart in range(n_restarts=3):
                 │
                 └─ _pgd_bipolar(z_batch, ids_batch, mask_batch,
                                 style_anchors, anti_anchors, v_semantic, cfg, encoder)
                      │                                   ↑ SEE DETAIL BELOW
                      └─ returns z_pos (500, 768), z_neg (500, 768)
                      └─ V_B = normalize(mean(z_pos - z_neg, dim=0))  → (768,)
                 │
                 └─ V_B_run = normalize(mean(restart_directions))     → (768,)
            │
            └─ all_directions.append(V_B_run)
       │
       └─ delta_star = normalize(mean(all_directions))   → (768,)
          torch.save(delta_star, cache/delta_star.pt)
```

### `_pgd_bipolar` — detailed call stack

This is the innermost loop. For one anchor batch it runs TWO PGD optimizations (positive pole + negative pole) and returns the final embeddings at each pole.

```
_pgd_bipolar(z, input_ids, attention_mask, style_anchors, anti_anchors, v_semantic, cfg, encoder, device)
  │
  ├─ [SETUP — no grad]
  │    encoder.get_intermediate_features(input_ids, attention_mask)
  │    ┌──────────────────────────────────────────────────────────────┐
  │    │  _get_embeddings(input_ids)                                  │
  │    │    └─ backbone.embeddings(input_ids) → (B, L, 768) embeds   │  FROZEN BODY
  │    │  extended_mask = backbone.get_extended_attention_mask(...)   │  (no grad)
  │    │  for layer in backbone.encoder.layer[:11]:                   │
  │    │    hidden = layer(hidden, extended_mask)[0]                  │
  │    └─ returns h_layer10: (B, L, 768)  ← intermediate repr        │
  │                                                                    ┘
  │    encoder.encode_with_delta_from_hidden(h_layer10, mask, zeros)
  │    ┌──────────────────────────────────────────────────────────────┐
  │    │  delta_full = zeros injected at CLS only (mask trick)        │  1-LAYER TAIL
  │    │  perturbed = h_layer10 + delta_full                          │  (no grad)
  │    │  for layer in backbone.encoder.layer[11:]:                   │
  │    │    perturbed = layer(perturbed, extended_mask)[0]            │
  │    │  cls = perturbed[:, 0, :]                                    │
  │    └─ returns z_orig: (B, 768) normalized CLS ← L_s reference    │
  │                                                                    ┘
  │
  └─ _run_pole(push_toward_style=True)   ← positive pole δ+
  └─ _run_pole(push_toward_style=False)  ← negative pole δ−
       │
       _run_pole(push_toward_style):
       │
       ├─ delta = zeros(B, 768).requires_grad_(True)   ← the variable being optimized
       ├─ optimizer = Adam([delta], lr=0.01)
       │
       └─ for step in range(n_steps=50):
            │
            ├─ encoder.encode_with_delta_from_hidden(h_layer10, mask, delta)
            │  ┌──────────────────────────────────────────────────────────────┐
            │  │  delta_full = delta broadcast to (B, L, 768), zeroed        │
            │  │               outside CLS position                           │
            │  │  perturbed = h_layer10 + delta_full                          │
            │  │               ↑ h_layer10 is DETACHED — no grad here        │
            │  │               ↑ delta_full carries grad from delta           │
            │  │  for layer in backbone.encoder.layer[11:]:  ← 1 layer only  │  GRAD GRAPH
            │  │    perturbed = layer(perturbed, extended_mask)[0]            │  (delta only)
            │  │  cls = perturbed[:, 0, :]                                    │
            │  └─ returns z_pert: (B, 768) normalized CLS                    │
            │                                                                  ┘
            │
            ├─ losses.l_bias_contrastive(z_pert, style_anchors, anti_anchors, push_toward_style)
            │    ├─ sim_style = z_pert @ style_anchors.T    (B, K)
            │    ├─ sim_anti  = z_pert @ anti_anchors.T     (B, K)
            │    ├─ logits = stack([sim_style, sim_anti], dim=-1).view(B*K, 2) * 100
            │    └─ L_B = cross_entropy(logits, target=[0]*B*K)  if push_toward_style
            │             cross_entropy(logits, target=[1]*B*K)  otherwise
            │    [Gradient: adaptive, contrastive, K-anchor — mirrors CLIP mix_pairs]
            │
            ├─ losses.l_semantic_preservation(z_pert, z_orig, v_semantic)
            │    ├─ delta_z = z_pert - z_orig.detach()       (B, 768)
            │    ├─ proj = (delta_z * v_semantic).sum(dim=-1) (B,)
            │    └─ returns (proj**2).mean()    ← penalize drift along semantic axis
            │
            ├─ loss = L_B + lambda_s * L_s
            ├─ loss.backward()      ← grad flows: loss → layer11 → delta_full → delta
            ├─ optimizer.step()     ← update delta
            └─ delta.data.clamp_(-epsilon, epsilon)   ← L∞ projection
       │
       └─ [FINAL FORWARD — no grad]
            encoder.encode_with_delta_from_hidden(h_layer10, mask, delta.detach())
            └─ returns z_final: (B, 768)  ← embedding at optimized pole
```

**Gradient flow summary:**

```
delta (B, 768)
  ↓  unsqueeze + expand + mask  →  delta_full (B, L, 768), nonzero at CLS only
  ↓  add to h_layer10 (detached)
  ↓  BertLayer 11 forward (self-attention + feed-forward)
  ↓  CLS extraction → normalize
  ↓  l_bias_contrastive(z_pert, style_anchors, anti_anchors) → L_B
     [cross-entropy over B*K pairwise logits; gradient adapts with confidence]
  ↓  + lambda_s * l_semantic_preservation → L_total
  ↓  backward
  ↑  gradient reaches delta only — no gradient accumulates in backbone weights
     (backbone weights are frozen: requires_grad=False)
```

---

## Phase 4 — `pipeline/clean.py`

**Purpose:** Project out the style direction from all cached tweet embeddings.

```
clean.py :: main()
  │
  ├─ for direction in [delta_star, v_style, v_shift, label_guided]:
  │    load direction vector (768,)
  │
  └─ apply_cleaning(direction, direction_name)
       └─ for split in [train, val, test]:
            load z_tweet_{split}.pt  → z (N, 768)
            │
            project_out(z, direction)
              ├─ d = normalize(direction)                      (768,)
              ├─ proj_scalar = (z @ d).unsqueeze(-1)          (N, 1)   ← scalar projection
              ├─ proj_vec    = proj_scalar * d.unsqueeze(0)   (N, 768) ← vector along d
              ├─ z_clean     = z - proj_vec                   (N, 768) ← orthogonal component
              └─ return normalize(z_clean)                    (N, 768)
            │
            torch.save({embeddings: z_clean, labels}, z_tweet_{split}_clean_{direction}.pt)
```

---

## Phase 5 — `pipeline/classify.py`

**Purpose:** Train a linear probe for each experimental condition and record F1.

```
classify.py :: main()
  │
  ├─ for condition in [B1_raw, B2_sae, B2.5_shift, B3_cbdc, C_oracle]:
  │    load z_tweet_train_clean_{direction}.pt  → X_train (N, 768), y_train
  │    load z_tweet_val_clean_{direction}.pt    → X_val,   y_val
  │    load z_tweet_test_clean_{direction}.pt   → X_test,  y_test
  │
  │    LinearProbe.__init__(input_dim=768, n_classes=3)
  │      └─ Linear(768, 3)   ← single layer, no hidden
  │
  │    train_probe(probe, X_train, y_train, X_val, y_val)
  │      for epoch in range(max_epochs):
  │        ├─ logits = probe(X_batch)             (B, 3)
  │        ├─ loss   = CrossEntropyLoss(logits, y)
  │        ├─ loss.backward(); optimizer.step()
  │        └─ early_stop on val macro-F1
  │
  └─  evaluate(probe, X_test, y_test) → macro F1, classification report
      torch.save(all_results, cache/results.pt)
```

---

## Phase 6 — `pipeline/evaluate.py`

**Purpose:** Interpretability checks + final report.

```
evaluate.py :: main()
  │
  ├─ load cache/results.pt
  ├─ print F1 table across conditions
  │
  ├─ direction_interpretability()
  │    for each class c in {negative, neutral, positive}:
  │      mean_proj[c] = mean(z_tweet[labels==c] @ direction)
  │    ← should be low variance across classes (direction is style, not sentiment)
  │
  ├─ linearity_check()
  │    for alpha in linspace(-1, 1):
  │      z_interp = normalize(z_formal_centroid + alpha * direction)
  │      cosine(z_interp, z_tweet_centroid)  ← should be monotonic if direction is linear
  │
  ├─ zero_shot_preservation()
  │    project_out(z_formal, direction)
  │    cosine(z_formal_clean, z_formal).mean()  ← should stay high (~1.0)
  │
  └─ write results/eval_report_new.txt
```

---

## How SAE and CBDC Interact — Data Flow

The two components operate in the **same 768-dim final CLS space** but at different pipeline stages:

```
                            768-dim final CLS embedding space
                            ─────────────────────────────────

Phase 2–3a  (SAE)          Phase 3b  (CBDC PGD)             Phase 4  (clean.py)
─────────────────          ────────────────────             ──────────────────
Trains dictionary          Uses style_anchors/anti_anchors  Projects delta_star
  W_d: 1536 features         as K paired poles in L_B:        out of embeddings
                               cross_entropy([z·s_i, z·a_i])
Scores features by                                           z_clean = z - (z·d)d
tweet vs formal            Finds delta_star such that:
activation delta             ∙ aligns with v_style     ──►  delta_star ∈ 768-dim
                               (diagnostic check)             CLS space
top-K decoder cols           ∙ doesn't drift along    ──►  same space as z_clean
  → style_anchors              v_semantic
bot-K decoder cols ───────►  PAIRED ANCHORS TO CBDC   ──►  USED HERE
  → anti_anchors                    │
        │                           │
v_style (aggregated)  ──────► DIAGNOSTICS ONLY
  sum of top-32                (cosine check post-run)
  decoder columns
```

**Key relationship:** `style_anchors` / `anti_anchors` (SAE output) serve as the K-anchor contrastive targets for the bipolar PGD L_B. They are the NLP analog of CLIP's `mix_pairs` — a set of distinct directions instead of a single fixed vector. Without the SAE, CBDC would have no definition of "tweet style" and would require contrastive text prompts like the original CLIP version.

`delta_star` (CBDC output) is a refined version of `v_style` that is:
- More precisely aligned to the tweet/formal axis (bipolar search removes noise)
- More orthogonal to `v_semantic` (semantic preservation loss enforces this)
- More stable across samples (averaged over 16 runs × 3 restarts)

---

## Cache Files — What Each Phase Reads and Writes

```
Phase 1 writes:   cache/z_tweet_train.pt   {embeddings, labels, input_ids, attention_mask}
                  cache/z_tweet_val.pt      {same}
                  cache/z_tweet_test.pt     {same}
                  cache/z_formal.pt         {embeddings}

Phase 2 reads:    cache/z_tweet_train.pt, cache/z_formal.pt
Phase 2 writes:   cache/sae_checkpoint.pt

Phase 3a reads:   cache/sae_checkpoint.pt, z_tweet_train.pt, z_formal.pt
Phase 3a writes:  cache/v_style.pt             (768,)   aggregated direction (diagnostics)
                  cache/v_shift.pt             (768,)   mean-shift baseline
                  cache/style_scores.pt        (1536,)  per-feature style scores
                  cache/style_anchors.pt       (K, 768) top-K decoder cols (tweet-differential)
                  cache/anti_style_anchors.pt  (K, 768) bot-K decoder cols (formal-differential)

Phase 3b reads:   cache/z_tweet_train.pt       (embeddings + input_ids + attention_mask)
                  cache/z_formal.pt            (embeddings only)
                  cache/v_style.pt             (diagnostics only)
                  cache/style_anchors.pt       (positive poles for L_B)
                  cache/anti_style_anchors.pt  (negative poles for L_B)
Phase 3b writes:  cache/delta_star.pt  (768,)

Phase 4 reads:    cache/z_tweet_{train,val,test}.pt
                  cache/delta_star.pt  (+ v_style, v_shift, computes label_guided)
Phase 4 writes:   cache/z_tweet_{split}_clean_{direction}.pt  — 4 × 3 = 12 files

Phase 5 reads:    all z_tweet_{split}_clean_{direction}.pt files
Phase 5 writes:   cache/results.pt

Phase 6 reads:    cache/results.pt, z_tweet_*.pt, z_formal.pt, direction vectors
Phase 6 writes:   results/eval_report_new.txt
```

---

## Backpropagation — Detailed Trace

### Setup

At the start of each `_pgd_bipolar` call, `h_layer10` is computed under `torch.no_grad()` and detached. Only `delta` has `requires_grad=True`. The gradient graph that `.backward()` builds is:

```
delta (B, 768)  requires_grad=True
  │
  ├─ unsqueeze(1) → expand(B, L, 768) → * mask  →  delta_full (B, L, 768)
  │   [mask is 1 only at CLS position 0; all other positions zero]
  │
  ├─ h_layer10 (detached, no grad) + delta_full  →  perturbed (B, L, 768)
  │   [h_layer10 is a constant tensor; grad flows only through delta_full]
  │
  └─ BertLayer 11 forward:
       ├─ Self-attention:
       │    Q = W_Q · perturbed   →  (B, L, H)
       │    K = W_K · perturbed   →  (B, L, H)
       │    V = W_V · perturbed   →  (B, L, H)
       │    scores = softmax( QK^T / √(H/heads) )  →  (B, heads, L, L)
       │    attn_out = scores · V  →  (B, L, H)
       │    out = W_O · attn_out  →  (B, L, H)
       │    x = LayerNorm( perturbed + out )  ← residual connection
       │
       ├─ FFN:
       │    intermediate = GELU( W_1 · x )  →  (B, L, 3072)
       │    x = LayerNorm( x + W_2 · intermediate )  ← residual
       │
       └─ perturbed = layer output  (B, L, 768)
  │
  cls = perturbed[:, 0, :]              (B, 768)   ← extract CLS position only
  z_pert = F.normalize(cls, dim=-1)     (B, 768)
  │
  ├─ L_B = l_bias_contrastive(z_pert, style_anchors, anti_anchors, push_toward_style)
  │         sim_style = z_pert @ style_anchors.T        (B, K)
  │         sim_anti  = z_pert @ anti_anchors.T         (B, K)
  │         logits    = stack([sim_style, sim_anti], -1).view(B*K, 2) * 100
  │         L_B = cross_entropy(logits, target=[0 or 1] * B*K)
  │
  └─ L_s = mean( ((z_pert - z_orig) · v_semantic)² )
            [z_orig is detached — treated as constant]
  │
  loss = L_B + lambda_s * L_s
  loss.backward()
```

### What each backward step computes

**∂loss/∂z_pert**
```
From L_B (cross-entropy over B*K pairwise logits):
  p_ij = softmax(100 * [z_pert[i] · s_j,  z_pert[i] · a_j])   for i=0..B-1, j=0..K-1
  ∂L_B/∂z_pert[i] = (100/B) * Σ_j (p_ij[0] - y_ij[0]) * s_j
                                   + (p_ij[1] - y_ij[1]) * a_j

  where y_ij = one-hot target (class 0 for positive pole, class 1 for negative pole).

  Properties:
  * Adaptive: (p_ij - y_ij) → 0 as the prediction becomes confident → gradient saturates
  * Contrastive: non-zero contribution from BOTH s_j and a_j in every step
  * Multi-anchor: sum over K pairs → weighted combination of K style directions

From L_s:  let proj_i = (z_pert[i] - z_orig[i]) · v_semantic
           ∂L_s/∂z_pert[i] = 2 * proj_i * v_semantic / B
           [adaptive — grows larger if z_pert has already drifted along v_semantic]
```

**∂z_pert/∂cls — L2 normalization Jacobian**
```
z_pert = cls / ‖cls‖
J = (I − z_pert z_pert^T) / ‖cls‖  ≈  I − z_pert z_pert^T

Effect: removes the radial component from the gradient.
Only the tangential component (perpendicular to z_pert on the unit sphere) reaches cls.
```

**∂cls/∂perturbed — CLS extraction**
```
cls = perturbed[:, 0, :]
Gradient is non-zero only at position 0 in the L dimension.
Positions 1..L-1 receive zero gradient from this step.
```

**∂perturbed_out/∂perturbed_in — BertLayer 11 Jacobian** ← key

Although only position 0's output is used, position 0's output in self-attention
is computed as a weighted sum of ALL L positions' values:

```
CLS output at layer 11 = Σ_j  attention_score(0→j)  ×  V[j]

where V[j] = W_V · perturbed[j]

Since perturbed[j] = h_layer10[j] for j > 0  (delta only affects j=0),
the gradient at delta reaches through two paths:

Path A — direct (CLS query/key/value):
  CLS output ← W_O · attn_scores_from_CLS · V
            ← delta affects Q[0], K[0], V[0] directly

Path B — attention re-weighting:
  CLS output ← attention_scores change because K[0] changed (delta changes K[0])
             → how much each other token j attends TO CLS changes
             → V-weighted sum changes

Result: gradient at delta encodes the full attention mixing geometry of layer 11,
not just a single CLS→CLS path. The gradient "sees" how CLS's relationship to
all tokens determines style alignment.
```

**∂perturbed_in/∂delta — injection**
```
perturbed[:, 0, :] = h_layer10[:, 0, :] + delta   (h_layer10 detached)
perturbed[:, j, :] = h_layer10[:, j, :]            for j > 0

∂perturbed[:, 0, :]/∂delta = I   (identity)
∂perturbed[:, j, :]/∂delta = 0   for j > 0
```

**Adam update**
```
g = delta.grad                    (B, 768) — aggregated gradient from all above
m ← β1·m + (1-β1)·g              momentum (direction smoothing)
v ← β2·v + (1-β2)·g²             adaptive scale (per-dimension)
delta ← delta - lr · m̂ / (√v̂ + ε)
delta.data.clamp_(-epsilon, epsilon)   L∞ projection
```

Adam's per-dimension scaling means components of `delta` where the gradient is
consistently small (orthogonal directions) get amplified less than the style-aligned
dimensions where the gradient is steep. This naturally shapes delta toward the style axis.

---

## CLIP vs NLP — Backprop Comparison

### Original CLIP: `perturb_bafa_txt_multi_ablation_lb_ls` in simple_pgd.py

```
z_adv1 (S, D_intermediate)   ← intermediate repr BEFORE projection
  │
  └─ target_model(z_adv1.half(), t_)  =  SetTarget.forward(z_adv1):
       ├─ for block in transformer.resblocks[-N:]:   ← last N blocks (default N=1)
       │    z_adv1 = block(z_adv1)
       ├─ ln_final(z_adv1)                           ← layer norm
       └─ z_adv1[token_max] @ text_projection        ← linear projection (D → D_proj)
       → adv_feat (B, D_proj)   normalized

  adv_feat
  │
  ├─ L_B (att_loss):
  │    logits[i,j] = [ adv_feat[i] · mix_a[j],  adv_feat[i] · mix_b[j] ]
  │    att_loss = cross_entropy( 100 * logits, label=[0,0,...] )
  │    [mix_a = bias prompts pole A, mix_b = bias prompts pole B]
  │    Gradient: ∂CE/∂adv_feat = Σ_j (softmax_j - one_hot_j) ⊗ [mix_a[j], mix_b[j]]
  │
  └─ L_s (keep_loss):
       keep_loss = 100 * ((adv_feat - ori) @ keep.T).pow(2).mean()
       [keep = neutral class text embeddings, ori = original clean embedding]
       loss = att_loss * (1 - keep_weight) - keep_loss * keep_weight

  loss.backward()
  grad_sign = z_adv1.grad.sign()
  z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)
  ← sign-SGD step (not Adam)
```

### Side-by-side comparison

| | CLIP original | NLP adaptation |
|---|---|---|
| **z lives in** | Pre-projection intermediate space (S×D, full sequence) | Layer-10 hidden state → perturbed at CLS only |
| **Forward tail** | Last N transformer blocks + `text_projection` (linear, D→D_proj) | `BertLayer[11]` only (no projection) |
| **Gradient path length** | N resblocks + 1 projection | 1 BertLayer |
| **L_B loss** | Cross-entropy over `adv_feat @ mix_pairs` (K text-pair anchors) | Cross-entropy over `z_pert @ [style_anchors, anti_anchors]` (K SAE-derived pairs) |
| **L_B gradient direction** | Σ (softmax − one_hot) ⊗ mix_pair cols → weighted combination of K bias text directions | Σ (softmax − one_hot) ⊗ [style_anchors[j], anti_anchors[j]] → weighted combination of K SAE decoder cols |
| **L_B gradient magnitude** | Adaptive — decreases as prediction becomes confident (softmax → 1) | Same — adaptive saturation via softmax |
| **L_s definition** | `((adv_feat − ori) @ keep.T)².mean()` where `keep` = semantic class texts | `((z_pert − z_orig) · v_semantic)².mean()` |
| **L_s sign in loss** | `loss = att_loss*(1-kw) − keep_loss*kw` → MAXIMIZES drift along keep directions | `loss = L_B + λ_s * L_s` → MINIMIZES drift along v_semantic |
| **Optimizer** | Sign-SGD (sign of gradient) + L∞ clamp | Adam + L∞ clamp |
| **Restarts** | `num_samples` loop with random perturbations | `n_restarts` loop |

### On the L_s sign difference

The CLIP version uses `-keep_loss * keep_weight` in the loss, which means it **maximizes**
the change along `keep` (semantic class) directions, not minimizes it. This is the opposite
of the NLP version's `+lambda_s * L_s`. This appears intentional: the CLIP version is
pushing the adversarial embedding AWAY from the semantic class directions to ensure the
found direction is style-specific and not class-specific. The NLP version instead penalizes
drift along the formal centroid direction, which acts as a weaker form of the same idea.

### On the L_B design: why cross-entropy with multi-anchor pairs

The original NLP adaptation used `L_B = ±cosine(z_pert, v_style)`, which had three structural gaps compared to CLIP:

| Gap | Cosine (old) | Cross-entropy + K anchors (current) |
|---|---|---|
| Gradient saturation | Fixed magnitude — never decays near target | Softmax: (p_j − y_j) → 0 as prediction becomes confident |
| Contrastive signal | Only pulls toward v_style; no push away from formal | Simultaneously pushes toward style class AND away from anti-style class |
| Anchor richness | Single fixed v_style vector | K SAE decoder columns (tweet-differential) paired with K anti-anchors (formal-differential) |

The current `l_bias_contrastive` in `losses.py` directly mirrors `perturb_bafa_txt_multi_ablation_lb_ls`:

```
CLIP:
  logits[i,j] = [adv_feat[i] · mix_a[j],   adv_feat[i] · mix_b[j]]
  mix_a = bias prompts (e.g. "he" variants), mix_b = debias prompts (e.g. "she" variants)

NLP:
  logits[i,j] = [z_pert[i]  · style_j,     z_pert[i]  · anti_j  ]
  style_j = top-K SAE decoder columns,  anti_j = bottom-K SAE decoder columns
```

Both give the same gradient structure: a weighted combination of K anchor directions, where
weights adapt with prediction confidence.

**Why not use RoPE-style rotation?**

A rotation-based relative objective (à la RoPE) could encode "style" as a relative angular
displacement rather than alignment to a fixed target. However, RoPE encodes sequence position
through rotation in 2D subspaces of the feature dimension — there is no principled analog
for "style direction" in the CBDC bipolar search. The K-anchor cross-entropy achieves
the desired adaptive, contrastive gradient without the rotational complexity, and maintains
a direct structural parallel to the CLIP implementation.

**On the v_style role change:**

`v_style` (the aggregated weighted sum of top-K decoder columns) is now only used for the
post-run diagnostic cosine check. It is no longer the fixed PGD target. The K individual
decoder columns (style_anchors) replace it as the contrastive anchors, which means:
- The PGD is no longer pulled toward a single point — it is guided by a K-dimensional
  subspace of tweet-style directions
- Each SAE feature contributes independently rather than being collapsed into one vector
- The gradient direction varies per anchor, creating more diverse V_B across runs
