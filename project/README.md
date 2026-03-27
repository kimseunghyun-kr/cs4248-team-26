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
  ├─ top_k_indices = argsort(style_scores, descending=True)[:32]
  │
  ├─ v_style = normalize(W_d[:, top_k_indices] @ style_scores[top_k_indices])
  │    └─ W_d[:, j]: the j-th decoder column = "what does feature j look like in embed space"
  │    └─ weighted sum of top-32 decoder columns → (768,) tweet-style direction
  │
  ├─ v_shift = normalize(mean(z_tweet) - mean(z_formal))  → (768,) naive baseline
  │
  └─ torch.save(v_style, cache/v_style.pt)
     torch.save(v_shift, cache/v_shift.pt)
```

**What `v_style` is:** A direction in the final 768-dim CLS embedding space that most separates tweet-like from formal-like content, as learned by the SAE's sparse dictionary. This is the CBDC paper's "bias direction candidate" — good but not yet purified.

---

## Phase 3b — `cbdc/refine.py`  ← CORE

**Purpose:** Refine `v_style` into `delta_star` using bipolar PGD. This is the CBDC algorithm.

```
refine.py :: main()
  │
  ├─ load z_tweet_train.pt  → z_tweet (N, 768), input_ids (N, L), attention_mask (N, L)
  ├─ load z_formal.pt       → z_formal (M, 768)
  ├─ load v_style.pt        → v_style (768,)
  ├─ FinBERTEncoder(model_name, device)
  │
  └─ collect_delta_star(z_tweet, input_ids, attention_mask, v_style, z_formal, cfg, encoder)
       │
       ├─ v_semantic = normalize(z_formal.mean(0))    → (768,)  formal centroid = semantic axis
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
                 └─ _pgd_bipolar(z_batch, ids_batch, mask_batch, v_style, v_semantic, cfg, encoder)
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
_pgd_bipolar(z, input_ids, attention_mask, v_style, v_semantic, cfg, encoder, device)
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
            ├─ L_B = -cosine(z_pert, v_style).mean()        if push_toward_style
            │        +cosine(z_pert, v_style).mean()         otherwise
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
  ↓  cosine(z_pert, v_style) → L_B
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
Trains dictionary          Uses v_style as PGD target       Projects delta_star
  W_d: 1536 features         └─ L_B = -cosine(z_pert,         out of embeddings
                                               v_style)
Scores features by                                           z_clean = z - (z·d)d
tweet vs formal            Finds delta_star such that:
activation delta             ∙ aligns with v_style     ──►  delta_star ∈ 768-dim
                             ∙ doesn't drift along            CLS space
v_style = weighted             v_semantic               ──►  same space as z_clean
  sum of top-32
  decoder columns  ───────►  INPUT TO CBDC              ──►  USED HERE
        │                          │
        └──────── v_style ─────────┘
                  (768,)
                  "what tweet style looks like in CLS space"
```

**Key relationship:** `v_style` (SAE output) serves as the attraction target for the positive pole of the bipolar PGD. Without the SAE, CBDC would have no starting definition of "tweet style" — it would need contrastive text prompts like the original CLIP version. The SAE replaces those prompts.

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
Phase 3a writes:  cache/v_style.pt  (768,)
                  cache/v_shift.pt  (768,)

Phase 3b reads:   cache/z_tweet_train.pt  (embeddings + input_ids + attention_mask)
                  cache/z_formal.pt       (embeddings only)
                  cache/v_style.pt
Phase 3b writes:  cache/delta_star.pt  (768,)

Phase 4 reads:    cache/z_tweet_{train,val,test}.pt
                  cache/delta_star.pt  (+ v_style, v_shift, computes label_guided)
Phase 4 writes:   cache/z_tweet_{split}_clean_{direction}.pt  — 4 × 3 = 12 files

Phase 5 reads:    all z_tweet_{split}_clean_{direction}.pt files
Phase 5 writes:   cache/results.pt

Phase 6 reads:    cache/results.pt, z_tweet_*.pt, z_formal.pt, direction vectors
Phase 6 writes:   results/eval_report_new.txt
```
