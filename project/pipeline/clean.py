"""
Phase 3: Apply orthogonal projection to clean cached embeddings.

Supports three projection types:
  - Single vector (H,):   z_clean = z - (z · d) * d
  - Subspace (K, H):      z_clean = z - (z @ U^T) @ U
  - Full matrix (H, H):   z_clean = z @ P^T   (debias_vl projection)

Run from project/ directory:
  python pipeline/clean.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------
def project_out(z: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Remove confound component from embeddings.

    Args:
        z:         (N, H) L2-normalized embeddings
        direction: (H,)   single direction vector
                   (K, H) orthonormal subspace rows
                   (H, H) full debiasing matrix (debias_vl)

    Returns:
        z_clean: (N, H) L2-normalized, confound component removed
    """
    if direction.dim() == 2 and direction.shape[0] == direction.shape[1]:
        # Full debiasing matrix (H, H) from debias_vl
        z_clean = z @ direction.to(z.device).T
    elif direction.dim() == 2:
        # Subspace (K, H)
        U = F.normalize(direction.to(z.device), dim=-1)
        proj = (z @ U.T) @ U
        z_clean = z - proj
    else:
        # Single vector (H,)
        d = F.normalize(direction.to(z.device), dim=-1)
        proj = (z @ d).unsqueeze(-1) * d.unsqueeze(0)
        z_clean = z - proj
    return F.normalize(z_clean, dim=-1)


# ---------------------------------------------------------------------------
# Sentiment boost: remove confound subspace, amplify sentiment subspace
# ---------------------------------------------------------------------------
def project_out_and_boost(
    z: torch.Tensor,
    confound_dirs: torch.Tensor,
    sentiment_prototypes: torch.Tensor,
    alpha: float = 2.0,
) -> torch.Tensor:
    """Remove confound subspace, then amplify the sentiment subspace.

    Args:
        z:              (N, H) embeddings
        confound_dirs:  (K, H) directions to remove (cbdc_directions)
        sentiment_prototypes:
                        (3, H) class prototypes [neg, neu, pos] or
                        (2, H) precomputed sentiment basis
        alpha:          amplification factor (>1 boosts sentiment signal)

    For 3-way sentiment classification, a 2D sentiment subspace is more faithful
    than a single positive-minus-negative axis. If class prototypes are given,
    we form two centered directions (pos-neu, neg-neu) and orthonormalize them.

    Returns:
        z_boost: (N, H) L2-normalized, confound removed + sentiment amplified
    """
    # Remove confound subspace
    U_conf = F.normalize(confound_dirs.to(z.device), dim=-1)
    z_clean = z - (z @ U_conf.T) @ U_conf

    # Build an orthonormal sentiment basis from either class prototypes or
    # a precomputed basis tensor.
    sentiment_prototypes = sentiment_prototypes.to(z.device)
    if sentiment_prototypes.dim() == 1:
        U_sent = F.normalize(sentiment_prototypes, dim=-1).unsqueeze(0)
    elif sentiment_prototypes.shape[0] == 3:
        neg, neu, pos = F.normalize(sentiment_prototypes, dim=-1)
        v1 = F.normalize(pos - neu, dim=-1)
        v2 = neg - neu
        v2 = v2 - torch.dot(v2, v1) * v1
        if v2.norm() < 1e-8:
            v2 = neg - pos
            v2 = v2 - torch.dot(v2, v1) * v1
        if v2.norm() < 1e-8:
            U_sent = v1.unsqueeze(0)
        else:
            v2 = F.normalize(v2, dim=-1)
            U_sent = torch.stack([v1, v2], dim=0)
    else:
        U_sent = F.normalize(sentiment_prototypes, dim=-1)

    sent_proj = (z_clean @ U_sent.T) @ U_sent
    z_boost = z_clean + (alpha - 1.0) * sent_proj

    return F.normalize(z_boost, dim=-1)


# ---------------------------------------------------------------------------
# Apply cleaning to all splits
# ---------------------------------------------------------------------------
def apply_cleaning(direction: torch.Tensor, direction_name: str) -> None:
    """Project out direction from all cached tweet splits."""
    print(f"\nApplying '{direction_name}' projection ...")

    for split in ["train", "val", "test"]:
        in_path = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
        out_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_clean_{direction_name}.pt")

        if not os.path.exists(in_path):
            print(f"  [skip] Missing {in_path}")
            continue

        data = torch.load(in_path, map_location="cpu")
        z = data["embeddings"]
        labels = data.get("labels")

        z_clean = project_out(z, direction, alpha=3.0)

        # Report magnitude removed
        if direction.dim() == 2 and direction.shape[0] == direction.shape[1]:
            diff = (z - z_clean).norm(dim=-1).mean().item()
            print(f"  {split}: mean L2 diff={diff:.4f} | shape={tuple(z_clean.shape)}")
        elif direction.dim() == 2:
            U = F.normalize(direction, dim=-1)
            proj_mag = (z @ U.T).abs().mean().item()
            print(f"  {split}: removed proj magnitude={proj_mag:.4f} | shape={tuple(z_clean.shape)}")
        else:
            d = F.normalize(direction, dim=-1)
            proj_mag = (z @ d).abs().mean().item()
            print(f"  {split}: removed proj magnitude={proj_mag:.4f} | shape={tuple(z_clean.shape)}")

        out_data = {"embeddings": z_clean}
        if labels is not None:
            out_data["labels"] = labels
        torch.save(out_data, out_path)
        print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Sentiment boost materialization
# ---------------------------------------------------------------------------
def materialize_sentiment_boost_conditions(alpha: float = 2.0) -> bool:
    """Create D4/D5 cached embeddings if the required artifacts exist."""
    sent_path = os.path.join(CACHE_DIR, "sentiment_prototypes.pt")
    cbdc_path = os.path.join(CACHE_DIR, "cbdc_directions.pt")

    if not os.path.exists(sent_path) or not os.path.exists(cbdc_path):
        missing = []
        if not os.path.exists(sent_path):
            missing.append("sentiment_prototypes.pt")
        if not os.path.exists(cbdc_path):
            missing.append("cbdc_directions.pt")
        print(f"\n[skip] Sentiment boost prerequisites missing: {', '.join(missing)}")
        return False

    sentiment_prototypes = torch.load(sent_path, map_location="cpu")
    confound_dirs = torch.load(cbdc_path, map_location="cpu")

    # D4: boost applied to raw embeddings after removing CBDC confounds.
    print(f"\nApplying sentiment boost (alpha={alpha}) on raw embeddings -> 'raw_sentiment_boost' ...")
    for split in ["train", "val", "test"]:
        in_path = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
        out_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_clean_raw_sentiment_boost.pt")
        if not os.path.exists(in_path):
            print(f"  [skip] Missing {in_path}")
            continue
        data = torch.load(in_path, map_location="cpu")
        z = data["embeddings"]
        z_boost = project_out_and_boost(z, confound_dirs, sentiment_prototypes, alpha=alpha)
        out_data = {"embeddings": z_boost}
        if "labels" in data:
            out_data["labels"] = data["labels"]
        torch.save(out_data, out_path)
        print(f"  {split}: boost shape={tuple(z_boost.shape)} -> {out_path}")

    # D5: boost applied to CBDC-encoded embeddings.
    cbdc_encoded = os.path.join(CACHE_DIR, "z_tweet_train_cbdc.pt")
    if os.path.exists(cbdc_encoded):
        print(f"\nApplying sentiment boost (alpha={alpha}) on CBDC embeddings -> 'cbdc_sentiment_boost' ...")
        for split in ["train", "val", "test"]:
            in_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_cbdc.pt")
            out_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_clean_cbdc_sentiment_boost.pt")
            if not os.path.exists(in_path):
                print(f"  [skip] Missing {in_path}")
                continue
            data = torch.load(in_path, map_location="cpu")
            z = data["embeddings"]
            z_boost = project_out_and_boost(z, confound_dirs, sentiment_prototypes, alpha=alpha)
            out_data = {"embeddings": z_boost}
            if "labels" in data:
                out_data["labels"] = data["labels"]
            torch.save(out_data, out_path)
            print(f"  {split}: boost shape={tuple(z_boost.shape)} -> {out_path}")
    else:
        print("\n[skip] CBDC-encoded embeddings missing; cannot build 'cbdc_sentiment_boost'.")

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    directions_to_run = []

    # debias_vl projection matrix
    dvl_path = os.path.join(CACHE_DIR, "debias_vl_P.pt")
    if os.path.exists(dvl_path):
        P = torch.load(dvl_path, map_location="cpu")
        directions_to_run.append((P, "debias_vl"))

    # CBDC directions (subspace)
    cbdc_path = os.path.join(CACHE_DIR, "cbdc_directions.pt")
    if os.path.exists(cbdc_path):
        dirs = torch.load(cbdc_path, map_location="cpu")
        directions_to_run.append((dirs, "cbdc_directions"))

    if not directions_to_run:
        print("No directions found. Run cbdc/refine.py first.")
        return

    for direction, name in directions_to_run:
        apply_cleaning(direction, name)

    # ---- Sentiment boost conditions (D4, D5) --------------------------------
    materialize_sentiment_boost_conditions(alpha=2.0)

    print("\nCleaning complete.")

# clean.py

# 1. ADD this function right below project_out()
def amplify_bias(z: torch.Tensor, direction: torch.Tensor, alpha: float = 3.0) -> torch.Tensor:
    """
    Amplifies the confound component to stretch the dataset along stylistic axes.
    alpha=3.0 applies a heavy multiplier to the bias direction.
    """
    if direction.dim() == 2 and direction.shape[0] == direction.shape[1]:
        # Full debias_vl matrix 
        return F.normalize(z @ direction.T, dim=-1) # Fallback to standard projection for full matrix
    
    if direction.dim() == 2:
        # Subspace projection (K, H)
        proj = (z @ direction.T) @ direction
        return F.normalize(z + (alpha * proj), dim=-1)
        
    elif direction.dim() == 1:
        # Single vector (H,)
        d = F.normalize(direction, dim=-1)
        proj = (z @ d).unsqueeze(-1) * d
        return F.normalize(z + (alpha * proj), dim=-1)
        
    return z


if __name__ == "__main__":
    main()
