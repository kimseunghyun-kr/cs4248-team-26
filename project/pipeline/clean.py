"""
Phase 4: Apply orthogonal projection to clean cached embeddings.

For a given direction d (unit vector), the cleaned embedding is:
    z_clean = L2_normalize(z - (z · d) * d)

This removes the component of z along d, leaving only the perpendicular component.
This is the core CBDC operation (paper eq. 2), adapted to pre-cached embeddings.

Run from project/ directory:
  python pipeline/clean.py --direction delta_star   # CBDC (B3)
  python pipeline/clean.py --direction v_style      # SAE only (B2)
  python pipeline/clean.py --direction v_shift      # mean-shift (B2.5)
  python pipeline/clean.py --direction label_guided # label-guided (C)

Outputs (saved to cache/):
  z_tweet_{split}_clean_{direction_name}.pt  for split in {train, val, test}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


# ---------------------------------------------------------------------------
# Core projection function
# ---------------------------------------------------------------------------
def project_out(z: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Remove the component of z along direction.

    Args:
        z         : (B, H)  L2-normalized embeddings
        direction : (H,)    unit direction vector

    Returns:
        z_clean   : (B, H)  L2-normalized, direction component removed
    """
    d = F.normalize(direction.to(z.device), dim=-1)           # (H,)
    proj_scalar = (z @ d).unsqueeze(-1)                        # (B, 1)
    proj_vec    = proj_scalar * d.unsqueeze(0)                 # (B, H)
    z_clean     = z - proj_vec                                 # (B, H)
    return F.normalize(z_clean, dim=-1)                        # (B, H) re-normalized


# ---------------------------------------------------------------------------
# Label-guided direction computation (Option C)
# ---------------------------------------------------------------------------
def compute_label_guided_direction(
    z_train: torch.Tensor,       # (N, H)
    labels_train: torch.Tensor,  # (N,) values in {0,1,2}
    z_formal: torch.Tensor,      # (M, H)
) -> torch.Tensor:
    """
    Compute the label-guided style direction.

    Approach: within-class mean-shift
      For each sentiment class c:
          v_c = mean(z_tweet[labels==c]) - mean(z_formal)
      Average across classes → removes class-correlated variance.
      Normalize → v_label_style.

    This is more principled than plain mean-shift because it finds the style
    component consistent within each sentiment class, not confounded by
    class-level sentiment signal.
    """
    formal_centroid = z_formal.mean(0)   # (H,)
    class_directions = []

    for c in [0, 1, 2]:
        mask = labels_train == c
        if mask.sum() == 0:
            continue
        class_mean = z_train[mask].mean(0)                    # (H,)
        v_c        = class_mean - formal_centroid              # (H,)
        class_directions.append(v_c)

    # Average within-class directions
    v_label_style = torch.stack(class_directions, dim=0).mean(0)   # (H,)
    return F.normalize(v_label_style, dim=-1)


# ---------------------------------------------------------------------------
# Apply cleaning to all splits
# ---------------------------------------------------------------------------
def apply_cleaning(direction: torch.Tensor, direction_name: str) -> None:
    """Project out 'direction' from all cached tweet splits and save."""
    direction = F.normalize(direction, dim=-1)
    print(f"\nApplying '{direction_name}' projection ...")

    for split in ["train", "val", "test"]:
        in_path  = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
        out_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_clean_{direction_name}.pt")

        if not os.path.exists(in_path):
            print(f"  ⚠ Missing {in_path}, skipping")
            continue

        data   = torch.load(in_path, map_location="cpu")
        z      = data["embeddings"]   # (N, H)
        labels = data.get("labels")

        z_clean = project_out(z, direction)

        # Report how much was removed
        proj_magnitude = (z @ direction.to(z.device)).abs().mean().item()
        print(f"  {split}: removed projection magnitude={proj_magnitude:.4f} "
              f"| shape={tuple(z_clean.shape)}")

        out_data = {"embeddings": z_clean}
        if labels is not None:
            out_data["labels"] = labels
        torch.save(out_data, out_path)
        print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direction",
        choices=["delta_star", "v_style", "v_shift", "label_guided", "all"],
        default="all",
        help="Which direction(s) to project out. 'all' runs all four.",
    )
    args = parser.parse_args()

    directions_to_run = (
        ["delta_star", "v_style", "v_shift", "label_guided"]
        if args.direction == "all"
        else [args.direction]
    )

    for d_name in directions_to_run:
        if d_name == "label_guided":
            # Compute label-guided direction from training data
            train_path  = os.path.join(CACHE_DIR, "z_tweet_train.pt")
            formal_path = os.path.join(CACHE_DIR, "z_formal.pt")

            if not os.path.exists(train_path) or not os.path.exists(formal_path):
                print(f"⚠ Skipping label_guided: missing cache files.")
                continue

            train_data = torch.load(train_path, map_location="cpu")
            z_train    = train_data["embeddings"]
            labels     = train_data["labels"]
            z_formal   = torch.load(formal_path, map_location="cpu")["embeddings"]

            direction = compute_label_guided_direction(z_train, labels, z_formal)
            print(f"Label-guided direction computed.")

        else:
            d_path = os.path.join(CACHE_DIR, f"{d_name}.pt")
            if not os.path.exists(d_path):
                print(f"⚠ Skipping '{d_name}': {d_path} not found.")
                continue
            direction = torch.load(d_path, map_location="cpu")

        apply_cleaning(direction, d_name)

    print("\nCleaning complete.")


if __name__ == "__main__":
    main()
