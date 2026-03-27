"""
Phase 3b: CBDC-PGD refinement → delta_star.

NLP adaptation of CBDC (CVPR 2026, Section 4.3–4.4) for FinBERT embeddings.

Loss design — aligned with CLIP's perturb_bafa_txt_multi_ablation_lb_ls:
  CLIP L_B: cross-entropy over pairwise logits [z·a_i, z·b_i] for each
    pair (a_i, b_i) from (bias_anchors, debias_anchors).  Gradient decays
    naturally near the target (softmax saturation), is contrastive (pushes
    away from the opposite class), and is multi-anchor.

  NLP L_B: same structure using SAE-derived anchor pairs:
    style_anchors[i]  — top-K SAE decoder columns (tweet-differential)
    anti_anchors[i]   — bottom-K SAE decoder columns (formal-differential)
    logits = [z_pert · style_i,  z_pert · anti_i]   for i = 0..K-1
    L_B = cross_entropy(100 * logits, target_class)

  This replaces the previous single cosine(z_pert, v_style) which had:
    * Fixed gradient magnitude — no saturation near target
    * No contrastive push away from formal side
    * Single anchor — less robust than K-anchor set

Bipolar PGD  (paper eq. 10 → Section 4.4):
  For each anchor batch from z_tweet_train:
    Positive pole — push toward tweet style:
      L_B+ = cross_entropy([z_pert · s_i, z_pert · a_i], target=0)  ← style class
      L_s+ =  || v_semantic · (z_pert+ − z_orig) ||²
      δ+ optimized to minimize  L_B+ + λ_s · L_s+
    Negative pole — push away from tweet style (toward formal):
      L_B− = cross_entropy([z_pert · s_i, z_pert · a_i], target=1)  ← anti-style class
      L_s− =  || v_semantic · (z_pert− − z_orig) ||²
      δ− optimized to minimize  L_B− + λ_s · L_s−
    Clean direction (paper eq. 6):
      V_B = normalize(mean(z_pert+ − z_pert−))

  delta_star = normalize(mean(V_B over all runs))

Structural features:
  * δ is injected at the CLS position of the layer-10 intermediate hidden
    state via encoder.encode_with_delta_from_hidden(h_layer10, mask, δ).
    Gradients flow through only BERT layer 11 (the 1-layer tail), matching
    the original CLIP RN50 route where PGD perturbs the representation
    just before attnpool/c_proj.
  * h_layer10 is pre-computed once per run under torch.no_grad() (frozen body),
    then reused across all PGD steps.
  * Each anchor batch uses different input_ids → different h_layer10 →
    different loss curvature per run → diversity in V_B.
  * v_style (SAE-derived aggregated direction) is used ONLY for diagnostics;
    the PGD loss uses style_anchors/anti_anchors (K individual decoder columns).

Prerequisites:
  sae_analysis.py must have been run to produce:
    cache/style_anchors.pt        — (top_k, H) tweet-differential SAE columns
    cache/anti_style_anchors.pt   — (top_k, H) formal-differential SAE columns
    cache/v_style.pt              — (H,) aggregated direction (diagnostics only)
  embed.py must save input_ids and attention_mask alongside embeddings:
    z_tweet_train.pt must contain keys: "embeddings", "labels",
                                        "input_ids", "attention_mask"

Outputs:
  cache/delta_star.pt  — (H,) refined bipolar style direction

Run from project/ directory:
  python cbdc/refine.py [--n_anchors 500] [--n_steps 50] [--n_runs 16]
                        [--n_restarts 3] [--epsilon 0.10] [--lambda_s 0.2]
                        [--n_style_anchors 8]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import PGDConfig
from encoder import FinBERTEncoder
from losses import l_bias_contrastive, l_semantic_preservation

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


# ---------------------------------------------------------------------------
# Bipolar PGD — gradient flows through BERT layer 11 (1-layer tail) only
# ---------------------------------------------------------------------------
def _pgd_bipolar(
    z: torch.Tensor,               # (B, H)  cached final embedding — used for L_s reference only
    input_ids: torch.Tensor,       # (B, L)  tokenized inputs for this batch
    attention_mask: torch.Tensor,  # (B, L)
    style_anchors: torch.Tensor,   # (K, H)  tweet-style anchor directions (positive poles)
    anti_anchors: torch.Tensor,    # (K, H)  formal-style anchor directions (negative poles)
    v_semantic: torch.Tensor,      # (H,)    formal centroid (semantic axis to preserve)
    cfg: PGDConfig,
    encoder: FinBERTEncoder,
    device: str,
    rand_eps: float = 0.0,         # random init radius (0 = zero init)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bipolar PGD: two opposing perturbations from the same anchor batch.

    Positive δ+: cross-entropy L_B pushes z_pert toward style_anchors (target=0)
    Negative δ−: cross-entropy L_B pushes z_pert toward anti_anchors (target=1)

    L_B uses K anchor pairs (style_anchors[i], anti_anchors[i]) — analogous to
    CLIP's mix_pairs, giving a contrastive, multi-anchor, saturation-aware gradient.

    Optimizer: sign-SGD with L∞ projection after each step (Madry et al.).
    This is the theoretically correct optimizer for L∞-bounded PGD — the original
    CLIP code uses it exclusively. Adam's adaptive scaling is unnecessary when every
    coordinate is clamped to [-ε, ε] regardless.

    Gradient path: loss → BERT layer 11 (1-layer tail) → h_layer10_CLS + δ → δ.
    This mirrors the CLIP RN50 route: z_adv perturbed before attnpool/c_proj.
    Backbone weights are frozen (requires_grad=False); only δ accumulates grads.

    Returns:
        z_pos  (B, H)  normalized embeddings at tweet pole
        z_neg  (B, H)  normalized embeddings at formal pole
        z_orig (B, H)  normalized embeddings at δ=0 (needed for v_plus/v_minus in caller)
    """
    B, H = z.shape
    z              = z.to(device).detach()
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    style_anchors  = F.normalize(style_anchors.to(device), dim=-1)   # (K, H)
    anti_anchors   = F.normalize(anti_anchors.to(device),  dim=-1)   # (K, H)
    v_semantic     = F.normalize(v_semantic.to(device), dim=-1)

    # Pre-compute intermediate hidden state (layers 0–10) once — frozen body.
    # h_layer10 is detached; gradient will only flow through layer 11.
    with torch.no_grad():
        h_layer10 = encoder.get_intermediate_features(input_ids, attention_mask)
        # z_orig: reference embedding with δ=0 for L_s computation and v_plus/v_minus.
        z_orig = encoder.encode_with_delta_from_hidden(
            h_layer10, attention_mask,
            torch.zeros(B, H, device=device)
        )  # (B, H)

    def _run_pole(push_toward_style: bool) -> torch.Tensor:
        # Random or zero initialization
        if rand_eps > 0.0:
            init = (torch.rand(B, H, device=device) * 2.0 - 1.0) * rand_eps
        else:
            init = torch.zeros(B, H, device=device)

        delta = init.clone().requires_grad_(True)

        for _ in range(cfg.n_steps):
            if delta.grad is not None:
                delta.grad.zero_()

            # Forward through layer 11 only: gradient flows to delta via CLS of h_layer10
            z_pert = encoder.encode_with_delta_from_hidden(
                h_layer10, attention_mask, delta
            )  # (B, H)

            # L_B: CLIP-style multi-anchor cross-entropy
            #   Positive pole → target class 0 (style)   pulls z_pert toward style_anchors
            #   Negative pole → target class 1 (anti)    pulls z_pert toward anti_anchors
            L_B = l_bias_contrastive(z_pert, style_anchors, anti_anchors, push_toward_style)

            # L_s: preserve projection onto sentiment semantic axis (paper eq. 9)
            L_s = l_semantic_preservation(z_pert, z_orig, v_semantic)

            (L_B + cfg.lambda_s * L_s).backward()

            with torch.no_grad():
                # Sign-SGD step: equal treatment of all dimensions (correct for L∞ PGD)
                delta.data -= cfg.step_lr * delta.grad.sign()
                # L∞ projection (paper eq. 10 / original clamper)
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)

        # Final forward pass with optimized delta (no grad)
        with torch.no_grad():
            z_final = encoder.encode_with_delta_from_hidden(
                h_layer10, attention_mask, delta.detach()
            )
        return z_final  # (B, H)

    z_pos = _run_pole(push_toward_style=True)
    z_neg = _run_pole(push_toward_style=False)
    return z_pos, z_neg, z_orig


# ---------------------------------------------------------------------------
# Collect delta_star over multiple runs
# ---------------------------------------------------------------------------
def collect_delta_star(
    z_tweet: torch.Tensor,               # (N, H)
    input_ids_tweet: torch.Tensor,       # (N, L)
    attention_mask_tweet: torch.Tensor,  # (N, L)
    labels_tweet: torch.Tensor,          # (N,)   sentiment labels (0=neg, 1=neu, 2=pos)
    style_anchors: torch.Tensor,         # (K, H)  tweet-style anchor directions
    anti_anchors: torch.Tensor,          # (K, H)  formal-style anchor directions
    v_style: torch.Tensor,               # (H,)    aggregated direction (diagnostics only)
    cfg: PGDConfig,
    encoder: FinBERTEncoder,
    n_runs: int,
    n_restarts: int,
    device: str,
    n_components: int = 3,
) -> torch.Tensor:
    """
    Collects bipolar directions over n_runs random anchor batches with n_restarts
    random initialisations each, then extracts a (K, H) style subspace via SVD.

    Sentiment axis (v_semantic):
      v_semantic = normalize(mean(z_tweet[pos]) − mean(z_tweet[neg]))
      This directly targets the sentiment axis for L_s preservation,
      rather than using the formal centroid as an indirect proxy.

    Per-restart collection (v_plus, v_minus separate):
      v_plus  = normalize(mean(z_pos − z_orig))   — style push direction
      v_minus = normalize(mean(z_orig − z_neg))   — formal push direction
      These are averaged per run, then stacked → (2*n_runs, H).

    Subspace SVD (CBDC paper Section 4.3):
      Mean-center the direction matrix, run torch.linalg.svd, return top-K
      right singular vectors as delta_subspace (K, H) — orthonormal rows.

    style_anchors / anti_anchors (K, H) replace the single v_style in L_B,
    giving a contrastive multi-anchor loss analogous to CLIP's mix_pairs.
    v_style is retained only for the post-run diagnostic cosine check.

    Returns delta_subspace: (n_components, H) orthonormal subspace basis.
    """
    # --- Sentiment axis -------------------------------------------------------
    pos_mask = (labels_tweet == 2)
    neg_mask = (labels_tweet == 0)
    z_pos_sent = z_tweet[pos_mask].mean(0).to(device)
    z_neg_sent = z_tweet[neg_mask].mean(0).to(device)
    v_semantic = F.normalize(z_pos_sent - z_neg_sent, dim=-1)

    # Sanity: cos(v_semantic, pos_centroid) ≈ 1, cos(v_semantic, neg_centroid) ≈ -1
    cos_sem_pos = F.cosine_similarity(v_semantic.unsqueeze(0),
                                      F.normalize(z_pos_sent.unsqueeze(0), dim=-1)).item()
    cos_sem_neg = F.cosine_similarity(v_semantic.unsqueeze(0),
                                      F.normalize(z_neg_sent.unsqueeze(0), dim=-1)).item()
    print(f"  v_semantic = sentiment axis  "
          f"cos(pos)={cos_sem_pos:.4f}  cos(neg)={cos_sem_neg:.4f}")
    print(f"  (pos_mask={pos_mask.sum().item()}  neg_mask={neg_mask.sum().item()})")
    print(f"  style_anchors: {tuple(style_anchors.shape)}  "
          f"anti_anchors: {tuple(anti_anchors.shape)}")

    batch_size = min(cfg.n_anchors, len(z_tweet))
    all_directions = []  # will hold (v_plus, v_minus) per run — 2 entries per run

    for run in tqdm(range(n_runs), desc="PGD bipolar runs"):
        indices    = torch.randperm(len(z_tweet))[:batch_size]
        z_batch    = z_tweet[indices]
        ids_batch  = input_ids_tweet[indices]
        mask_batch = attention_mask_tweet[indices]

        restart_pos = []   # v_plus  per restart
        restart_neg = []   # v_minus per restart
        rand_eps = cfg.epsilon * 0.5 if n_restarts > 1 else 0.0

        for restart in range(n_restarts):
            ep = rand_eps if restart > 0 else 0.0
            z_pos, z_neg, z_orig = _pgd_bipolar(
                z_batch, ids_batch, mask_batch,
                style_anchors, anti_anchors, v_semantic,
                cfg, encoder, device,
                rand_eps=ep,
            )
            with torch.no_grad():
                v_plus  = F.normalize((z_pos - z_orig).mean(dim=0), dim=-1)
                v_minus = F.normalize((z_orig - z_neg).mean(dim=0), dim=-1)
                restart_pos.append(v_plus.cpu())
                restart_neg.append(v_minus.cpu())

        # Average across restarts → one stable direction per pole per run
        run_plus  = F.normalize(torch.stack(restart_pos).mean(0), dim=-1)
        run_minus = F.normalize(torch.stack(restart_neg).mean(0), dim=-1)
        all_directions.append(run_plus)
        all_directions.append(run_minus)

    # --- Subspace SVD ---------------------------------------------------------
    directions   = torch.stack(all_directions, dim=0)          # (2*n_runs, H)
    directions_c = directions - directions.mean(dim=0)          # mean-center

    _, _, Vh = torch.linalg.svd(directions_c, full_matrices=False)
    delta_subspace = Vh[:n_components]                          # (K, H)  orthonormal rows

    # --- Diagnostics ----------------------------------------------------------
    print(f"\n  delta_subspace shape: {tuple(delta_subspace.shape)}")

    # Orthonormality check: delta_subspace @ delta_subspace.T ≈ I_K
    gram = delta_subspace @ delta_subspace.T
    eye  = torch.eye(n_components)
    ortho_err = (gram - eye).abs().max().item()
    print(f"  Orthonormality error (max |GG^T − I|): {ortho_err:.2e}  (should be ≈ 0)")

    for i in range(n_components):
        comp = delta_subspace[i]
        c_style = F.cosine_similarity(comp.unsqueeze(0), v_style.unsqueeze(0)).item()
        c_sem   = F.cosine_similarity(comp.unsqueeze(0), v_semantic.cpu().unsqueeze(0)).item()
        print(f"  component {i}: cosine(v_style)={c_style:.4f}  cosine(v_semantic)={c_sem:.4f}")

    if delta_subspace[0] @ v_style > 0.3:
        print("  ✓ CBDC subspace component_0 aligns with v_style")
    else:
        print("  ⚠ Low alignment — try more n_steps or larger epsilon")

    # Mean pairwise cosine for diversity check
    n_dirs = directions.shape[0]
    pairwise = directions @ directions.T   # (2*n_runs, 2*n_runs)
    off_diag = pairwise[~torch.eye(n_dirs, dtype=torch.bool)].mean().item()
    print(f"  mean pairwise cosine between directions = {off_diag:.4f}  (lower → more diverse)")

    return delta_subspace


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_anchors",       type=int,   default=500)
    parser.add_argument("--n_steps",         type=int,   default=50)
    parser.add_argument("--n_runs",          type=int,   default=16)
    parser.add_argument("--n_restarts",      type=int,   default=3,
                        help="Random restarts per run (mirrors num_samples in CLIP code).")
    parser.add_argument("--n_style_anchors", type=int,   default=8,
                        help="Number of SAE anchor pairs (style + anti-style) used in L_B. "
                             "Slices the first K rows from style_anchors.pt. "
                             "CLIP typically uses 4–8 pairs. Default: 8.")
    parser.add_argument("--epsilon",         type=float, default=0.10)
    parser.add_argument("--lambda_s",        type=float, default=0.2)
    parser.add_argument("--step_lr",         type=float, default=0.01)
    parser.add_argument("--n_components",    type=int,   default=3,
                        help="Number of SVD principal components for the style subspace. "
                             "Output delta_subspace.pt has shape (n_components, H). Default: 3.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load cached data ---------------------------------------------------
    tweet_path        = os.path.join(CACHE_DIR, "z_tweet_train.pt")
    vstyle_path       = os.path.join(CACHE_DIR, "v_style.pt")
    style_anch_path   = os.path.join(CACHE_DIR, "style_anchors.pt")
    anti_anch_path    = os.path.join(CACHE_DIR, "anti_style_anchors.pt")

    for p in [tweet_path, vstyle_path, style_anch_path, anti_anch_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\nRun embed.py → sae.py → sae_analysis.py first."
            )

    tweet_data = torch.load(tweet_path, map_location="cpu")
    z_tweet    = tweet_data["embeddings"]

    # input_ids, attention_mask and labels must have been saved by the updated embed.py
    for key in ("input_ids", "attention_mask", "labels"):
        if key not in tweet_data:
            raise KeyError(
                f"z_tweet_train.pt is missing '{key}'.\n"
                "Re-run embed.py with the updated version that saves token tensors and labels."
            )
    input_ids_tweet      = tweet_data["input_ids"]       # (N, L)
    attention_mask_tweet = tweet_data["attention_mask"]  # (N, L)
    labels_tweet         = tweet_data["labels"]           # (N,)  0=neg 1=neu 2=pos

    v_style  = torch.load(vstyle_path, map_location="cpu")

    # Load anchor matrices from sae_analysis.py; slice to --n_style_anchors
    style_anchors_full = torch.load(style_anch_path, map_location="cpu")  # (top_k, H)
    anti_anchors_full  = torch.load(anti_anch_path,  map_location="cpu")  # (top_k, H)
    K = min(args.n_style_anchors, style_anchors_full.shape[0])
    style_anchors = style_anchors_full[:K]   # (K, H)
    anti_anchors  = anti_anchors_full[:K]    # (K, H)

    print(f"z_tweet={tuple(z_tweet.shape)} | v_style={tuple(v_style.shape)}")
    print(f"input_ids={tuple(input_ids_tweet.shape)} | "
          f"attention_mask={tuple(attention_mask_tweet.shape)}")
    print(f"style_anchors={tuple(style_anchors.shape)} (using {K} of "
          f"{style_anchors_full.shape[0]} available)")

    # ---- Load encoder -------------------------------------------------------
    model_name = os.environ.get("MODEL_NAME", "ProsusAI/finbert")
    encoder = FinBERTEncoder(model_name=model_name, device=device)

    # ---- Build config -------------------------------------------------------
    cfg = PGDConfig(
        epsilon      = args.epsilon,
        n_steps      = args.n_steps,
        step_lr      = args.step_lr,
        lambda_s     = args.lambda_s,
        n_anchors    = args.n_anchors,
        n_directions = args.n_runs,
        device       = device,
    )

    print(f"\nRunning CBDC bipolar PGD refinement (encoder-based, multi-anchor L_B) ...")
    print(f"  n_runs={args.n_runs} | n_restarts={args.n_restarts} | "
          f"n_anchors={args.n_anchors} | n_steps={args.n_steps} | "
          f"K={K} anchors | ε={args.epsilon} | λ_s={args.lambda_s}")

    delta_subspace = collect_delta_star(
        z_tweet              = z_tweet,
        input_ids_tweet      = input_ids_tweet,
        attention_mask_tweet = attention_mask_tweet,
        labels_tweet         = labels_tweet,
        style_anchors        = style_anchors,
        anti_anchors         = anti_anchors,
        v_style              = v_style,
        cfg                  = cfg,
        encoder              = encoder,
        n_runs               = args.n_runs,
        n_restarts           = args.n_restarts,
        device               = device,
        n_components         = args.n_components,
    )

    out_path = os.path.join(CACHE_DIR, "delta_subspace.pt")
    torch.save(delta_subspace, out_path)
    print(f"\ndelta_subspace shape={tuple(delta_subspace.shape)} saved → {out_path}")
    print("CBDC bipolar refinement complete.")


if __name__ == "__main__":
    main()
