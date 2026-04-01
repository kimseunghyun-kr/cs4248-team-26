"""
Reusable projection utilities for condition-specific embedding materialization.

This module is no longer a standalone pipeline phase. The default runner now
materializes D1 / D2 / D3 directly inside `cbdc/refine.py` to avoid cross-
condition `.pt` contamination.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def project_out(z: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Remove a confound component from embeddings and re-normalize the result.

    Supported `direction` formats:
      (H,)   single direction vector
      (K,H)  orthonormal subspace rows
      (H,H)  full projection/debiasing matrix
    """
    if direction.dim() == 2 and direction.shape[0] == direction.shape[1]:
        z_clean = z @ direction.to(z.device).T
    elif direction.dim() == 2:
        U = F.normalize(direction.to(z.device), dim=-1)
        z_clean = z - (z @ U.T) @ U
    else:
        d = F.normalize(direction.to(z.device), dim=-1)
        z_clean = z - (z @ d).unsqueeze(-1) * d.unsqueeze(0)
    return F.normalize(z_clean, dim=-1)


def main():
    print(
        "pipeline/clean.py is no longer a standalone phase. "
        "Use cbdc/refine.py to materialize D1 / D2 / D3 condition artifacts."
    )


if __name__ == "__main__":
    main()
