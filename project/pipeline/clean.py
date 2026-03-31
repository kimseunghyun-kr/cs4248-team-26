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
    sentiment_dirs: torch.Tensor,
    alpha: float = 2.0,
) -> torch.Tensor:
    """Remove confound subspace, then amplify the sentiment subspace.

    Args:
        z:              (N, H) embeddings
        confound_dirs:  (K, H) directions to remove (cbdc_directions)
        sentiment_dirs: (3, H) class prototypes [neg, neu, pos] or
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
    sentiment_dirs = sentiment_dirs.to(z.device)
    if sentiment_dirs.dim() == 1:
        U_sent = F.normalize(sentiment_dirs, dim=-1).unsqueeze(0)
    elif sentiment_dirs.shape[0] == 3:
        neg, neu, pos = F.normalize(sentiment_dirs, dim=-1)
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
        U_sent = F.normalize(sentiment_dirs, dim=-1)

    sent_proj = (z_clean @ U_sent.T) @ U_sent
    z_boost = z_clean + (alpha - 1.0) * sent_proj

    return F.normalize(z_boost, dim=-1)


# ---------------------------------------------------------------------------
# Label-guided direction computation (oracle comparison)
# ---------------------------------------------------------------------------
def compute_label_guided_direction(
    z_train: torch.Tensor,
    labels_train: torch.Tensor,
    z_formal: torch.Tensor,
) -> torch.Tensor:
    """Within-class mean-shift: average (class_mean - formal_mean) across classes."""
    formal_centroid = z_formal.mean(0)
    class_directions = []
    for c in [0, 1, 2]:
        mask = labels_train == c
        if mask.sum() == 0:
            continue
        v_c = z_train[mask].mean(0) - formal_centroid
        class_directions.append(v_c)
    v_label = torch.stack(class_directions).mean(0)
    return F.normalize(v_label, dim=-1)


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

        z_clean = project_out(z, direction)

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

    # Label-guided (oracle)
    train_path = os.path.join(CACHE_DIR, "z_tweet_train.pt")
    formal_path = os.path.join(CACHE_DIR, "z_formal.pt")
    if os.path.exists(train_path) and os.path.exists(formal_path):
        train_data = torch.load(train_path, map_location="cpu")
        z_formal = torch.load(formal_path, map_location="cpu")["embeddings"]
        direction = compute_label_guided_direction(
            train_data["embeddings"], train_data["labels"], z_formal
        )
        directions_to_run.append((direction, "label_guided"))
        print("Label-guided direction computed.")

    if not directions_to_run:
        print("No directions found. Run cbdc/refine.py first.")
        return

    for direction, name in directions_to_run:
        apply_cleaning(direction, name)

    # ---- Sentiment boost conditions (D4, D5) --------------------------------
    sent_path = os.path.join(CACHE_DIR, "sentiment_dirs.pt")
    cbdc_path = os.path.join(CACHE_DIR, "cbdc_directions.pt")
    if os.path.exists(sent_path) and os.path.exists(cbdc_path):
        sentiment_dirs = torch.load(sent_path, map_location="cpu")
        confound_dirs  = torch.load(cbdc_path, map_location="cpu")
        alpha = 2.0

        # D4: boost applied to raw embeddings using CBDC confound dirs.
        print(f"\nApplying sentiment boost (alpha={alpha}) on raw embeddings -> 'debias_vl_boost' ...")
        for split in ["train", "val", "test"]:
            in_path  = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
            out_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_clean_debias_vl_boost.pt")
            if not os.path.exists(in_path):
                print(f"  [skip] Missing {in_path}")
                continue
            data = torch.load(in_path, map_location="cpu")
            z = data["embeddings"]
            z_boost = project_out_and_boost(z, confound_dirs, sentiment_dirs, alpha=alpha)
            out_data = {"embeddings": z_boost}
            if "labels" in data:
                out_data["labels"] = data["labels"]
            torch.save(out_data, out_path)
            print(f"  {split}: boost shape={tuple(z_boost.shape)} -> {out_path}")

        # D5: boost applied to CBDC-encoded embeddings
        cbdc_encoded = os.path.join(CACHE_DIR, "z_tweet_train_cbdc.pt")
        if os.path.exists(cbdc_encoded):
            print(f"\nApplying sentiment boost (alpha={alpha}) on CBDC embeddings -> 'cbdc_boost' ...")
            for split in ["train", "val", "test"]:
                in_path  = os.path.join(CACHE_DIR, f"z_tweet_{split}_cbdc.pt")
                out_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_clean_cbdc_boost.pt")
                if not os.path.exists(in_path):
                    print(f"  [skip] Missing {in_path}")
                    continue
                data = torch.load(in_path, map_location="cpu")
                z = data["embeddings"]
                z_boost = project_out_and_boost(z, confound_dirs, sentiment_dirs, alpha=alpha)
                out_data = {"embeddings": z_boost}
                if "labels" in data:
                    out_data["labels"] = data["labels"]
                torch.save(out_data, out_path)
                print(f"  {split}: boost shape={tuple(z_boost.shape)} -> {out_path}")

    print("\nCleaning complete.")


if __name__ == "__main__":
    main()
