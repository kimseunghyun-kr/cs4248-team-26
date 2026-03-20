"""
Phase 3b: CBDC-PGD refinement of v_style → delta_star.

NLP adaptation of CBDC (CVPR 2026, Section 4.3–4.4) for FinBERT embeddings.

Key structural difference from the original CLIP implementation:
  CLIP uses a contrastive (cross-entropy) bias loss L_B against opposing text
  prompt embeddings (e.g. "he" vs "she").  In the NLP/FinBERT setting there is
  no paired image encoder, so we replace that with cosine alignment to the
  SAE-extracted style direction v_style (tweet corpus) and its negation
  (formal corpus direction).

Bipolar PGD  (paper eq. 10 → Section 4.4):
  For each anchor batch from z_tweet_train:
    Positive pole — push toward tweet style:
      L_B+ = -cosine(z + δ+, v_style)
      L_s+ =  || v_semantic · (z_pert+ − z) ||²
      δ+ optimized to minimize  L_B+ + λ_s · L_s+
    Negative pole — push away from tweet style (toward formal):
      L_B− = +cosine(z + δ−, v_style)      [same sign flip as −cosine to -v_style]
      L_s− =  || v_semantic · (z_pert− − z) ||²
      δ− optimized to minimize  L_B− + λ_s · L_s−
    Clean direction (paper eq. 6):
      V_B = normalize(mean(z_pert+ − z_pert−))

  delta_star = normalize(mean(V_B over all runs))

Reasonable compromises vs. CLIP version:
  * No encoder re-pass per PGD step — perturbation is applied directly to
    final CLS embeddings (valid per Section 4.2: "latent-space PGD operates
    on z = Ψ(x) directly").
  * v_style (SAE-derived) replaces the contrastive text prompt pairs used
    in CLIP; v_semantic = formal centroid replaces the neutral concept v_C.
  * Single aggregated delta_star replaces the full subspace S_B; clean.py
    applies a single orthogonal projection (sufficient for our linear-probe
    evaluation pipeline).
  * Random restarts within each run (n_restarts) mirror the num_samples
    loop in perturb_bafa_txt_multi_ablation_lb_ls from simple_pgd.py.

Outputs:
  cache/delta_star.pt  — (H,) refined bipolar style direction

Run from project/ directory:
  python cbdc/refine.py [--n_anchors 500] [--n_steps 50] [--n_runs 16]
                        [--n_restarts 3] [--epsilon 0.10] [--lambda_s 0.2]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import PGDConfig
from losses import l_semantic_preservation

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


# ---------------------------------------------------------------------------
# Bipolar PGD on pre-cached embeddings
# ---------------------------------------------------------------------------
def _pgd_bipolar(
    z: torch.Tensor,           # (B, H) — L2-normalized anchor embeddings
    v_style: torch.Tensor,     # (H,)   — tweet-style direction (positive pole)
    v_semantic: torch.Tensor,  # (H,)   — formal centroid (semantic axis to preserve)
    cfg: PGDConfig,
    device: str,
    rand_eps: float = 0.0,     # optional random initialization radius
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bipolar PGD: run two opposing perturbations from the same anchor batch.

    Positive δ+: aligns z+δ with v_style  (tweet pole)
    Negative δ−: aligns z+δ with −v_style (formal pole)

    Semantic preservation (L_s, paper eq. 9) is applied to both poles.

    Returns:
        z_pos (B, H)  — L2-normalized perturbed embeddings at tweet pole
        z_neg (B, H)  — L2-normalized perturbed embeddings at formal pole
    """
    z         = z.to(device).detach()
    v_style   = F.normalize(v_style.to(device),   dim=-1)
    v_semantic = F.normalize(v_semantic.to(device), dim=-1)

    def _run_pole(push_toward_style: bool) -> torch.Tensor:
        """Run PGD for one pole. Returns z_pert (B, H)."""
        if rand_eps > 0.0:
            init = (torch.rand_like(z) * 2 - 1) * rand_eps
        else:
            init = torch.zeros_like(z)

        delta = init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=cfg.step_lr)

        for _ in range(cfg.n_steps):
            optimizer.zero_grad()
            z_pert = F.normalize(z + delta, dim=-1)

            # L_B: align with (+v_style) for tweet pole, (−v_style) for formal pole
            cos = F.cosine_similarity(z_pert, v_style.unsqueeze(0), dim=-1)
            L_B = -cos.mean() if push_toward_style else cos.mean()

            # L_s: preserve projection onto formal semantic axis (paper eq. 9)
            L_s = l_semantic_preservation(z_pert, z, v_semantic)

            loss = L_B + cfg.lambda_s * L_s
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)

        return F.normalize((z + delta.detach()), dim=-1)   # (B, H)

    z_pos = _run_pole(push_toward_style=True)
    z_neg = _run_pole(push_toward_style=False)
    return z_pos, z_neg


# ---------------------------------------------------------------------------
# Collect delta_star over multiple runs (with optional random restarts)
# ---------------------------------------------------------------------------
def collect_delta_star(
    z_tweet: torch.Tensor,    # (N, H)
    v_style: torch.Tensor,    # (H,)
    z_formal: torch.Tensor,   # (M, H)
    cfg: PGDConfig,
    n_runs: int,
    n_restarts: int,
    device: str,
) -> torch.Tensor:
    """
    Collects bipolar directions V_B = normalize(mean(z_pos − z_neg)) over
    n_runs random anchor batches, each with n_restarts random initialisations.

    This mirrors the num_samples loop in perturb_bafa_txt_multi_ablation_lb_ls
    (simple_pgd.py).

    Returns delta_star: (H,) normalized aggregated direction.
    """
    # Semantic axis: L2-normalized formal centroid  (= v_C in the paper)
    v_semantic = F.normalize(z_formal.mean(0), dim=-1)
    print(f"  v_semantic = formal centroid  norm={v_semantic.norm():.4f}")

    batch_size = min(cfg.n_anchors, len(z_tweet))

    # Each element is one V_B direction (H,)
    all_directions = []

    for run in tqdm(range(n_runs), desc="PGD bipolar runs"):
        indices  = torch.randperm(len(z_tweet))[:batch_size]
        z_batch  = z_tweet[indices]   # (B, H)

        # --- multiple restarts per batch (mirrors num_samples) ---------------
        restart_directions = []
        rand_eps = cfg.epsilon * 0.5 if n_restarts > 1 else 0.0

        for restart in range(n_restarts):
            ep = rand_eps if restart > 0 else 0.0
            z_pos, z_neg = _pgd_bipolar(
                z_batch, v_style, v_semantic, cfg, device, rand_eps=ep
            )

            with torch.no_grad():
                # V_B = v+ − v−  (paper eq. 6, clean bias direction)
                V_B = (z_pos - z_neg).mean(dim=0)   # (H,)
                V_B = F.normalize(V_B, dim=-1)
                restart_directions.append(V_B.cpu())

        # Average across restarts for this run → one stable direction per run
        V_B_run = F.normalize(
            torch.stack(restart_directions, dim=0).mean(dim=0), dim=-1
        )
        all_directions.append(V_B_run)

    directions  = torch.stack(all_directions, dim=0)   # (n_runs, H)
    delta_star  = F.normalize(directions.mean(dim=0), dim=-1)

    # --- Alignment diagnostics -----------------------------------------------
    cos_with_vstyle = F.cosine_similarity(
        delta_star.unsqueeze(0), v_style.unsqueeze(0)
    ).item()
    cos_with_vsem = F.cosine_similarity(
        delta_star.unsqueeze(0), v_semantic.cpu().unsqueeze(0)
    ).item()
    print(f"\n  cosine(delta_star, v_style)    = {cos_with_vstyle:.4f}  (should be > 0.3)")
    print(f"  cosine(delta_star, v_semantic) = {cos_with_vsem:.4f}   (should be small)")

    # Report per-direction variance (diversity of discovered directions)
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
    parser.add_argument("--n_anchors",   type=int,   default=500)
    parser.add_argument("--n_steps",     type=int,   default=50)
    parser.add_argument("--n_runs",      type=int,   default=16)
    parser.add_argument("--n_restarts",  type=int,   default=3,
                        help="Random restarts per run (mirrors num_samples in CLIP code).")
    parser.add_argument("--epsilon",     type=float, default=0.10)
    parser.add_argument("--lambda_s",    type=float, default=0.2)
    parser.add_argument("--step_lr",     type=float, default=0.01)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load cached data ---------------------------------------------------
    tweet_path  = os.path.join(CACHE_DIR, "z_tweet_train.pt")
    formal_path = os.path.join(CACHE_DIR, "z_formal.pt")
    vstyle_path = os.path.join(CACHE_DIR, "v_style.pt")

    for p in [tweet_path, formal_path, vstyle_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}\nRun embed.py → sae.py → sae_analysis.py first.")

    z_tweet  = torch.load(tweet_path,  map_location="cpu")["embeddings"]
    z_formal = torch.load(formal_path, map_location="cpu")["embeddings"]
    v_style  = torch.load(vstyle_path, map_location="cpu")

    print(f"z_tweet={tuple(z_tweet.shape)} | z_formal={tuple(z_formal.shape)} | "
          f"v_style={tuple(v_style.shape)}")

    cfg = PGDConfig(
        epsilon      = args.epsilon,
        n_steps      = args.n_steps,
        step_lr      = args.step_lr,
        lambda_s     = args.lambda_s,
        n_anchors    = args.n_anchors,
        n_directions = args.n_runs,
        device       = device,
    )

    print(f"\nRunning CBDC bipolar PGD refinement ...")
    print(f"  n_runs={args.n_runs} | n_restarts={args.n_restarts} | "
          f"n_anchors={args.n_anchors} | n_steps={args.n_steps} | "
          f"ε={args.epsilon} | λ_s={args.lambda_s}")

    delta_star = collect_delta_star(
        z_tweet    = z_tweet,
        v_style    = v_style,
        z_formal   = z_formal,
        cfg        = cfg,
        n_runs     = args.n_runs,
        n_restarts = args.n_restarts,
        device     = device,
    )

    out_path = os.path.join(CACHE_DIR, "delta_star.pt")
    torch.save(delta_star, out_path)
    print(f"\ndelta_star saved → {out_path}")
    print("CBDC bipolar refinement complete.")


if __name__ == "__main__":
    main()
