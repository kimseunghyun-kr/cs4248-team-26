# Technical Reference

Algorithm internals for the 3-way tweet sentiment pipeline
(`negative / neutral / positive`) built around `debias_vl` and the
RN50-style `text_iccv` CBDC loop.

## Combined Pipeline: debias_vl → PGD → text_iccv

```
run_all.py :: main()
  └─ for each phase: subprocess.run([sys.executable, script_path])
       ├─ Phase 1  data/embed.py       → z_tweet_{split}.pt, z_formal.pt
       ├─ Phase 2  cbdc/refine.py      → debias_vl_P.pt, cbdc_directions.pt,
       │                                  sentiment_prototypes.pt, encoder_cbdc.pt,
       │                                  z_*_cbdc.pt, z_*_clean_cbdc_proj.pt
       ├─ Phase 3  pipeline/clean.py   → z_*_clean_{debias_vl,cbdc_directions,
       │                                  label_guided,raw_sentiment_boost,cbdc_sentiment_boost}.pt
       ├─ Phase 4  pipeline/classify.py → results.pt
       └─ Phase 5  pipeline/evaluate.py → results/eval_report.txt
```

## Phase 2: Combined debias_vl + CBDC (`cbdc/refine.py`)

Prompt roles in the current pipeline:

```
cls_text_groups   = class prototype bank for [negative, neutral, positive]
target_text       = 3 class-conditioned prompts attacked by PGD
keep_text         = neutral finance prompts used only for L_s
candidate_prompt  = sentiment x mined-topic crossed prompts for debias_vl
spurious_prompt   = mined topic/style prompts for debias_vl
```

### Phase A: debias_vl confound map discovery

Adapted from `references/debias_vl.py`. Uses a prompt bank mined from the
cached tweet train split when available, with static fallback topics otherwise.
The mined topic bank is saved to `prompt_bank.json`.

```
Input:
  spurious_cb  = encode(32 pure topic prompts)         # (32, H)
  candidate_cb = encode(96 sentiment x topic prompts)  # (96, H)
  S_pairs = 1488 same-sentiment, different-topic pairs
  B_pairs = 96 same-topic, different-sentiment pairs

Algorithm:
  P0 = I - V(V^T V)^{-1} V^T           # orthogonal complement of spurious subspace
  M  = avg over S_pairs of get_A(z_i, z_j)  # semantic preservation
  G  = 1000 * M + I                      # regularized
  P_debias = P0 @ G^{-1}                # (H, H) debiasing projection

  confound_matrix = I - P_debias
  U, S, Vh = svd(confound_matrix)
  svd_dirs = Vh[:K]                      # top-K confound directions
  bias_anchors = avg(top-scoring spurious prompts per svd_dir)
  anti_anchors = avg(bottom-scoring spurious prompts per svd_dir)
```

### Phase B+C: CBDC text_iccv loop

Adapted from `references/base.py::text_iccv`. Matches the original RN50 algorithm.

```
for epoch in range(100):

  # 1. Freeze layer 11, get target_text intermediate features
  h_target = frozen_body(target_text_ids)      # (3, L, H)
  z_orig = layer_11(h_target, delta=0)         # (3, H)

  # 2. Bipolar PGD on target_text, preserving keep_text semantics
  for restart in range(10):
    delta = 0 or random_init
    for step in range(20):
      z_pert = layer_11(h_target + delta_at_CLS)
      L_B = cross_entropy(100 * [z·bias_i, z·anti_i], target=0)
      L_s = 100 * ((z_pert - z_orig) @ keep_cb.T)^2.mean()   # MULTI-AXIS
      loss = L_B * (1 - 0.92) - L_s * 0.92
      delta += 0.0037 * sign(grad(loss))
      clamp(delta, -1.0, 1.0)                                 # L-inf
    adv_pos.append(z_pert)
    # repeat for negative pole (target=1)
    adv_neg.append(z_pert)

  # 3. Unfreeze layer 11, compute pooled class prototypes with gradient
  S = cat(adv_pos) - cat(adv_neg)
  cls_em = pool(encode_with_grad(cls_text_groups))  # (3, H)

  # 4. RN50-style class-specific match_loss
  for c in [0, 1, 2]:
    match_loss += (S[c::3] @ cls_em[c:c+1].T)^2.mean()
  match_loss *= 100

  # 5. ck_loss
  ck_loss = (bias_anchors - anti_anchors) @ cls_em.T)^2.mean() * 100

  # 6. Update layer 11
  (match_loss + ck_loss).backward()
  optimizer.step()
```

## Checkpoint Selection

The current code uses different selection rules in Phase 2 and Phase 4.

```
Phase 2 / cbdc/refine.py:
  - cbdc_directions.pt        = best-epoch SVD(best_S)
  - sentiment_prototypes.pt   = class prototypes from the best encoder
  - encoder_cbdc.pt           = best validation-selected layer-11 weights

Phase 4 / pipeline/classify.py:
  - each linear probe keeps the best validation checkpoint
  - reported val_f1 is the best validation F1 seen during probe training
```

Phase 2 now selects the encoder checkpoint with a lightweight validation
selector and then recomputes the final CBDC directions from that best
checkpoint.

## Gradient Flow

```
CLIP RN50 (original):                   FinBERT (adaptation):
  frozen ResNet body                       frozen BERT layers 0-10
          |                                        |
  intermediate repr                         h_layer10 (B, L, 768)
          |                                        |
       delta + z                            delta + h_CLS
          |                                        |
  attnpool + c_proj (trainable)            BERT layer 11 (trainable)
          |                                        |
       z_pert                                   z_pert
          |                                        |
  L_B + L_s (PGD step)                     L_B + L_s (PGD step)
  match_loss + ck_loss (train)              match_loss + ck_loss (train)
```

## Key Hyperparameters

| Parameter | Value | Original (RN50) | Description |
|-----------|-------|------------------|-------------|
| epsilon | 1.0 | att_bnd=1.0 | L-inf perturbation bound |
| n_pgd_steps | 20 | att_itr=20 | PGD iterations |
| step_lr | 0.0037 | att_stp=0.0037 | Sign-SGD step size |
| keep_weight | 0.92 | keep_weight=0.92 | L_s weight in PGD |
| num_samples | 10 | num_sam=10 | PGD restarts |
| n_epochs | 100 | txt_iters=100+ | Training epochs |
| up_scale | 100 | up_=100 | Loss multiplier |
| n_bias_dirs | 4 | — | SVD components from debias_vl |
| lambda_reg | 1000 | 1000 (debias_vl.py) | Regularization in G |

## Cache Dependencies

```
Phase 1 → z_tweet_{train,val,test}.pt, z_formal.pt
Phase 2 → debias_vl_P.pt, cbdc_directions.pt, sentiment_prototypes.pt,
           encoder_cbdc.pt, prompt_bank.json, anchor_poles.json,
           z_*_cbdc.pt, z_*_clean_cbdc_proj.pt
Phase 3 → z_*_clean_{debias_vl,cbdc_directions,label_guided,
           raw_sentiment_boost,cbdc_sentiment_boost}.pt
Phase 4 → results.pt
Phase 5 → eval_report.txt
```

## Experiment Conditions

```
B1 (raw)              raw tweet embeddings
D1 (debias_vl)        debias_vl projection on raw embeddings
D2 (CBDC)             embeddings from the CBDC-trained encoder
D3 (CBDC+proj)        CBDC embeddings + residual CBDC projection
D4 (raw+sent-boost)   raw embeddings + confound removal + sentiment boost
D5 (CBDC+sent-boost)  CBDC embeddings + confound removal + sentiment boost
C  (label-guided)     oracle-style label-guided projection baseline
```

`sentiment_prototypes.pt` stores the 3 pooled class prototypes. In Phase 3,
the boost path builds a 2D sentiment subspace from those prototypes
(`pos-neu` and `neg-neu`) rather than using a single `pos-neg` axis.
If D4/D5 embeddings are missing when Phase 4 starts, `classify.py` now
tries to materialize them from the Phase 2 artifacts automatically.
