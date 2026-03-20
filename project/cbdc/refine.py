"""
Phase 3b: CBDC-PGD refinement of v_style → delta_star.

Unlike the original direction_finder.py (which perturbs through the frozen
transformer), this operates directly in the pre-cached CLS embedding space.
This is valid because the CBDC paper's latent-space PGD operates on z = Ψ(x)
directly (section 4.2), without re-encoding through the transformer at each step.

Algorithm:
  For each anchor batch from z_tweet_train:
    1. Initialize δ = 0
    2. For n_steps iterations:
         z_pert = L2_normalize(z + δ)
         L_B    = -cosine_sim(z_pert, v_style)      # push toward tweet style
         L_s    = || v_semantic · (z_pert - z) ||²  # preserve formal semantics
         loss   = L_B + lambda_s * L_s
         δ ← Adam step on loss
         δ ← clip(δ, -ε, ε)   # ℓ∞ projection
    3. direction_i = normalize(mean(z_pert - z))
  delta_star = normalize(mean(all direction_i))

Outputs:
  cache/delta_star.pt  — (768,) refined style direction

Run from project/ directory:
  python cbdc/refine.py [--n_anchors 500] [--n_steps 50] [--n_runs 16]
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
# Core PGD loop (operates directly on cached embeddings — no encoder needed)
# ---------------------------------------------------------------------------
def _pgd_on_embeddings(
    z: torch.Tensor,          # (B, H) — L2-normalized anchor embeddings
    v_target: torch.Tensor,   # (H,)   — direction to push toward (v_style)
    v_semantic: torch.Tensor, # (H,)   — semantic axis to preserve (formal centroid)
    cfg: PGDConfig,
    device: str,
) -> torch.Tensor:
    """
    Runs PGD on a batch of pre-cached embeddings.
    Returns the per-sample delta* of shape (B, H).
    """
    z         = z.to(device).detach()
    v_target  = F.normalize(v_target.to(device), dim=-1)
    v_semantic = F.normalize(v_semantic.to(device), dim=-1)

    B, H = z.shape
    delta = torch.zeros(B, H, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.step_lr)

    for _ in range(cfg.n_steps):
        optimizer.zero_grad()

        z_pert = F.normalize(z + delta, dim=-1)   # (B, H)

        # L_B: maximize cosine similarity with v_target (tweet-style direction)
        L_B = -F.cosine_similarity(z_pert, v_target.unsqueeze(0), dim=-1).mean()

        # L_s: penalize deviation from formal semantic axis
        L_s = l_semantic_preservation(z_pert, z, v_semantic)

        loss = L_B + cfg.lambda_s * L_s
        loss.backward()
        optimizer.step()

        # ℓ∞ projection
        with torch.no_grad():
            delta.data.clamp_(-cfg.epsilon, cfg.epsilon)

    return delta.detach()


# ---------------------------------------------------------------------------
# Collect delta_star over multiple runs
# ---------------------------------------------------------------------------
def collect_delta_star(
    z_tweet: torch.Tensor,    # (N, H)
    v_style: torch.Tensor,    # (H,)
    z_formal: torch.Tensor,   # (M, H)
    cfg: PGDConfig,
    n_runs: int,
    device: str,
) -> torch.Tensor:
    """
    Runs PGD n_runs times on random sub-batches of z_tweet.
    Returns delta_star: (H,) normalized aggregated direction.
    """
    # Semantic preservation axis = L2-normalized formal centroid
    v_semantic = F.normalize(z_formal.mean(0), dim=-1)
    print(f"  v_semantic = formal centroid  norm={v_semantic.norm():.4f}")

    batch_size = min(cfg.n_anchors, len(z_tweet))
    all_directions = []

    for run in tqdm(range(n_runs), desc="PGD runs"):
        # Random sub-batch of anchors
        indices = torch.randperm(len(z_tweet))[:batch_size]
        z_batch = z_tweet[indices]   # (B, H)

        delta_star_batch = _pgd_on_embeddings(z_batch, v_style, v_semantic, cfg, device)

        with torch.no_grad():
            z_pert = F.normalize(z_batch.to(device) + delta_star_batch, dim=-1)
            z_orig = F.normalize(z_batch.to(device), dim=-1)
            direction = (z_pert - z_orig).mean(dim=0)     # (H,)
            direction = F.normalize(direction, dim=-1)
            all_directions.append(direction.cpu())

    directions = torch.stack(all_directions, dim=0)   # (n_runs, H)
    delta_star = F.normalize(directions.mean(dim=0), dim=-1)

    # Report alignment stats
    cos_with_vstyle = F.cosine_similarity(delta_star.unsqueeze(0), v_style.unsqueeze(0)).item()
    cos_with_vsem   = F.cosine_similarity(delta_star.unsqueeze(0), v_semantic.cpu().unsqueeze(0)).item()
    print(f"\n  cosine(delta_star, v_style)   = {cos_with_vstyle:.4f}  (should be > 0.3)")
    print(f"  cosine(delta_star, v_semantic) = {cos_with_vsem:.4f}  (should be small)")
    if cos_with_vstyle > 0.3:
        print("  ✓ CBDC refinement preserved alignment with v_style")
    else:
        print("  ⚠ Low alignment — try more n_steps or larger epsilon")

    return delta_star


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_anchors", type=int,   default=500)
    parser.add_argument("--n_steps",   type=int,   default=50)
    parser.add_argument("--n_runs",    type=int,   default=16)
    parser.add_argument("--epsilon",   type=float, default=0.10)
    parser.add_argument("--lambda_s",  type=float, default=0.2)
    parser.add_argument("--step_lr",   type=float, default=0.01)
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

    print(f"\nRunning CBDC-PGD refinement ...")
    print(f"  n_runs={args.n_runs} | n_anchors={args.n_anchors} | "
          f"n_steps={args.n_steps} | ε={args.epsilon} | λ_s={args.lambda_s}")

    delta_star = collect_delta_star(
        z_tweet  = z_tweet,
        v_style  = v_style,
        z_formal = z_formal,
        cfg      = cfg,
        n_runs   = args.n_runs,
        device   = device,
    )

    out_path = os.path.join(CACHE_DIR, "delta_star.pt")
    torch.save(delta_star, out_path)
    print(f"\ndelta_star saved → {out_path}")
    print("CBDC refinement complete.")


if __name__ == "__main__":
    main()
