# Technical Reference

Algorithm internals for the CBDC Financial Sentiment Pipeline.

## Combined Pipeline: debias_vl → PGD → text_iccv

```
run_all.py :: main()
  └─ for each phase: subprocess.run([sys.executable, script_path])
       ├─ Phase 1  data/embed.py       → z_tweet_{split}.pt, z_formal.pt
       ├─ Phase 2  cbdc/refine.py      → debias_vl_P.pt, cbdc_directions.pt,
       │                                  encoder_cbdc.pt, z_*_cbdc.pt, z_*_clean_cbdc_proj.pt
       ├─ Phase 3  pipeline/clean.py   → z_*_clean_{debias_vl,cbdc_directions,label_guided}.pt
       ├─ Phase 4  pipeline/classify.py → results.pt
       └─ Phase 5  pipeline/evaluate.py → results/eval_report.txt
```

## Phase 2: Combined debias_vl + CBDC (`cbdc/refine.py`)

### Phase A: debias_vl confound map discovery

Adapted from `references/debias_vl.py`. Uses (sentiment x topic) crossed prompts.

```
Input:
  spurious_cb  = encode(6 pure topic prompts)        # (6, H)
  candidate_cb = encode(18 sentiment x topic prompts) # (18, H)
  S_pairs = 45 same-sentiment, different-topic pairs
  B_pairs = 18 same-topic, different-sentiment pairs

Algorithm:
  P0 = I - V(V^T V)^{-1} V^T           # orthogonal complement of spurious subspace
  M  = avg over S_pairs of get_A(z_i, z_j)  # semantic preservation
  G  = 1000 * M + I                      # regularized
  P_debias = P0 @ G^{-1}                # (H, H) debiasing projection

  confound_matrix = I - P_debias
  U, S, Vh = svd(confound_matrix)
  bias_anchors = Vh[:K]                  # top-K confound directions
  anti_anchors = -Vh[:K]                 # opposing pole
```

### Phase B+C: CBDC text_iccv loop

Adapted from `references/base.py::text_iccv`. Matches the original RN50 algorithm.

```
for epoch in range(100):

  # 1. Freeze layer 11, get test_text intermediate features
  h_test = frozen_body(test_text_ids)          # (6, L, H)
  z_orig = layer_11(h_test, delta=0)           # (6, H)

  # 2. Bipolar PGD on test_text (6 neutral concept prompts)
  for restart in range(10):
    delta = 0 or random_init
    for step in range(20):
      z_pert = layer_11(h_test + delta_at_CLS)
      L_B = cross_entropy(100 * [z·bias_i, z·anti_i], target=0)
      L_s = 100 * ((z_pert - z_orig) @ test_cb.T)^2.mean()   # MULTI-AXIS
      loss = L_B * (1 - 0.92) - L_s * 0.92
      delta -= 0.0037 * sign(grad(loss))
      clamp(delta, -1.0, 1.0)                                 # L-inf
    adv_pos.append(z_pert)
    # repeat for negative pole (target=1)
    adv_neg.append(z_pert)

  # 3. Unfreeze layer 11, compute class embeddings with gradient
  S = cat(adv_pos) - cat(adv_neg)
  cls_em = encode_with_grad(cls_text)          # (3, H)

  # 4. Class-specific match_loss
  for c in [0, 1, 2]:
    match_loss += (S @ cls_em[c:c+1].T)^2.mean()
  match_loss *= 100

  # 5. ck_loss
  ck_loss = (bias_anchors - anti_anchors) @ cls_em.T)^2.mean() * 100

  # 6. Update layer 11
  (match_loss + ck_loss).backward()
  optimizer.step()
```

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
Phase 2 → debias_vl_P.pt, cbdc_directions.pt,
           encoder_cbdc.pt, z_*_cbdc.pt, z_*_clean_cbdc_proj.pt
Phase 3 → z_*_clean_{debias_vl,label_guided}.pt
Phase 4 → results.pt
Phase 5 → eval_report.txt
```
