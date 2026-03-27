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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bipolar PGD: two opposing perturbations from the same anchor batch.

    Positive δ+: cross-entropy L_B pushes z_pert toward style_anchors (target=0)
    Negative δ−: cross-entropy L_B pushes z_pert toward anti_anchors (target=1)

    L_B uses K anchor pairs (style_anchors[i], anti_anchors[i]) — analogous to
    CLIP's mix_pairs, giving a contrastive, multi-anchor, saturation-aware gradient.

    Gradient path: loss → BERT layer 11 (1-layer tail) → h_layer10_CLS + δ → δ.
    This mirrors the CLIP RN50 route: z_adv perturbed before attnpool/c_proj.
    Backbone weights are frozen (requires_grad=False); only δ accumulates grads.

    Returns:
        z_pos (B, H)  normalized embeddings at tweet pole
        z_neg (B, H)  normalized embeddings at formal pole
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
        # z_orig: reference embedding with δ=0 for L_s computation.
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
        optimizer = torch.optim.Adam([delta], lr=cfg.step_lr)

        for _ in range(cfg.n_steps):
            optimizer.zero_grad()

            # Forward through layer 11 only: gradient flows to delta via CLS of h_layer10
            z_pert = encoder.encode_with_delta_from_hidden(
                h_layer10, attention_mask, delta
            )  # (B, H)

            # L_B: CLIP-style multi-anchor cross-entropy
            #   Positive pole → target class 0 (style)   pulls z_pert toward style_anchors
            #   Negative pole → target class 1 (anti)    pulls z_pert toward anti_anchors
            L_B = l_bias_contrastive(z_pert, style_anchors, anti_anchors, push_toward_style)

            # L_s: preserve projection onto formal semantic axis (paper eq. 9)
            L_s = l_semantic_preservation(z_pert, z_orig, v_semantic)

            (L_B + cfg.lambda_s * L_s).backward()
            optimizer.step()

            # L∞ projection (paper eq. 10 / original clamper)
            with torch.no_grad():
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)

        # Final forward pass with optimized delta (no grad)
        with torch.no_grad():
            z_final = encoder.encode_with_delta_from_hidden(
                h_layer10, attention_mask, delta.detach()
            )
        return z_final  # (B, H)

    z_pos = _run_pole(push_toward_style=True)
    z_neg = _run_pole(push_toward_style=False)
    return z_pos, z_neg


# ---------------------------------------------------------------------------
# Collect delta_star over multiple runs
# ---------------------------------------------------------------------------
def collect_delta_star(
    z_tweet: torch.Tensor,               # (N, H)
    input_ids_tweet: torch.Tensor,       # (N, L)
    attention_mask_tweet: torch.Tensor,  # (N, L)
    style_anchors: torch.Tensor,         # (K, H)  tweet-style anchor directions
    anti_anchors: torch.Tensor,          # (K, H)  formal-style anchor directions
    v_style: torch.Tensor,               # (H,)    aggregated direction (diagnostics only)
    z_formal: torch.Tensor,              # (M, H)
    cfg: PGDConfig,
    encoder: FinBERTEncoder,
    n_runs: int,
    n_restarts: int,
    device: str,
) -> torch.Tensor:
    """
    Collects bipolar directions V_B = normalize(mean(z_pos − z_neg)) over
    n_runs random anchor batches, each with n_restarts random initialisations.

    style_anchors / anti_anchors (K, H) replace the single v_style in L_B,
    giving a contrastive multi-anchor loss analogous to CLIP's mix_pairs.
    v_style is retained only for the post-run diagnostic cosine check.

    Returns delta_star: (H,) normalized aggregated direction.
    """
    v_semantic = F.normalize(z_formal.mean(0), dim=-1)
    print(f"  v_semantic = formal centroid  norm={v_semantic.norm():.4f}")
    print(f"  style_anchors: {tuple(style_anchors.shape)}  "
          f"anti_anchors: {tuple(anti_anchors.shape)}")

    batch_size = min(cfg.n_anchors, len(z_tweet))
    all_directions = []

    for run in tqdm(range(n_runs), desc="PGD bipolar runs"):
        indices    = torch.randperm(len(z_tweet))[:batch_size]
        z_batch    = z_tweet[indices]
        ids_batch  = input_ids_tweet[indices]
        mask_batch = attention_mask_tweet[indices]

        restart_directions = []
        rand_eps = cfg.epsilon * 0.5 if n_restarts > 1 else 0.0

        for restart in range(n_restarts):
            ep = rand_eps if restart > 0 else 0.0
            z_pos, z_neg = _pgd_bipolar(
                z_batch, ids_batch, mask_batch,
                style_anchors, anti_anchors, v_semantic,
                cfg, encoder, device,
                rand_eps=ep,
            )
            with torch.no_grad():
                # V_B = v+ − v− (paper eq. 6)
                V_B = F.normalize((z_pos - z_neg).mean(dim=0), dim=-1)
                restart_directions.append(V_B.cpu())

        # Average across restarts → one stable direction per run
        V_B_run = F.normalize(
            torch.stack(restart_directions, dim=0).mean(dim=0), dim=-1
        )
        all_directions.append(V_B_run)

    directions = torch.stack(all_directions, dim=0)   # (n_runs, H)
    delta_star = F.normalize(directions.mean(dim=0), dim=-1)

    # --- Diagnostics ----------------------------------------------------------
    # v_style is the aggregated (weighted-sum) direction from sae_analysis.py;
    # a good delta_star should still align with it even though we no longer
    # fix the PGD target to this single vector.
    cos_with_vstyle = F.cosine_similarity(
        delta_star.unsqueeze(0), v_style.unsqueeze(0)
    ).item()
    cos_with_vsem = F.cosine_similarity(
        delta_star.unsqueeze(0), v_semantic.cpu().unsqueeze(0)
    ).item()
    print(f"\n  cosine(delta_star, v_style)    = {cos_with_vstyle:.4f}  (should be > 0.3)")
    print(f"  cosine(delta_star, v_semantic) = {cos_with_vsem:.4f}   (should be small)")

    # Mean cosine of delta_star to each individual style anchor
    cos_per_anchor = (style_anchors @ delta_star.unsqueeze(-1)).squeeze(-1)  # (K,)
    print(f"  cosine(delta_star, style_anchors) — mean={cos_per_anchor.mean():.4f}  "
          f"max={cos_per_anchor.max():.4f}  min={cos_per_anchor.min():.4f}")

    # Pairwise diversity: lower = more diverse runs
    pairwise = directions @ directions.T   # (n_runs, n_runs)
    off_diag = pairwise[~torch.eye(n_runs, dtype=torch.bool)].mean().item()
    print(f"  mean pairwise cosine between runs = {off_diag:.4f}  (lower → more diverse)")

    if cos_with_vstyle > 0.3:
        print("  ✓ CBDC bipolar refinement preserved alignment with v_style")
    else:
        print("  ⚠ Low alignment — try more n_steps or larger epsilon")

    return delta_star


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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load cached data ---------------------------------------------------
    tweet_path        = os.path.join(CACHE_DIR, "z_tweet_train.pt")
    formal_path       = os.path.join(CACHE_DIR, "z_formal.pt")
    vstyle_path       = os.path.join(CACHE_DIR, "v_style.pt")
    style_anch_path   = os.path.join(CACHE_DIR, "style_anchors.pt")
    anti_anch_path    = os.path.join(CACHE_DIR, "anti_style_anchors.pt")

    for p in [tweet_path, formal_path, vstyle_path, style_anch_path, anti_anch_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\nRun embed.py → sae.py → sae_analysis.py first."
            )

    tweet_data = torch.load(tweet_path, map_location="cpu")
    z_tweet    = tweet_data["embeddings"]

    # input_ids and attention_mask must have been saved by the updated embed.py
    if "input_ids" not in tweet_data or "attention_mask" not in tweet_data:
        raise KeyError(
            "z_tweet_train.pt is missing 'input_ids' / 'attention_mask'.\n"
            "Re-run embed.py with the updated version that saves token tensors."
        )
    input_ids_tweet      = tweet_data["input_ids"]       # (N, L)
    attention_mask_tweet = tweet_data["attention_mask"]  # (N, L)

    z_formal = torch.load(formal_path, map_location="cpu")["embeddings"]
    v_style  = torch.load(vstyle_path, map_location="cpu")

    # Load anchor matrices from sae_analysis.py; slice to --n_style_anchors
    style_anchors_full = torch.load(style_anch_path, map_location="cpu")  # (top_k, H)
    anti_anchors_full  = torch.load(anti_anch_path,  map_location="cpu")  # (top_k, H)
    K = min(args.n_style_anchors, style_anchors_full.shape[0])
    style_anchors = style_anchors_full[:K]   # (K, H)
    anti_anchors  = anti_anchors_full[:K]    # (K, H)

    print(f"z_tweet={tuple(z_tweet.shape)} | z_formal={tuple(z_formal.shape)} | "
          f"v_style={tuple(v_style.shape)}")
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

    delta_star = collect_delta_star(
        z_tweet              = z_tweet,
        input_ids_tweet      = input_ids_tweet,
        attention_mask_tweet = attention_mask_tweet,
        style_anchors        = style_anchors,
        anti_anchors         = anti_anchors,
        v_style              = v_style,
        z_formal             = z_formal,
        cfg                  = cfg,
        encoder              = encoder,
        n_runs               = args.n_runs,
        n_restarts           = args.n_restarts,
        device               = device,
    )

    out_path = os.path.join(CACHE_DIR, "delta_star.pt")
    torch.save(delta_star, out_path)
    print(f"\ndelta_star saved → {out_path}")
    print("CBDC bipolar refinement complete.")


if __name__ == "__main__":
    main()
