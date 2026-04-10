from __future__ import annotations

import torch
import torch.nn.functional as F


def project_out(z: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Remove a component or subspace from embeddings and re-normalize.

    Supported `direction` formats:
      (H,)   single direction vector
      (K,H)  orthonormal subspace rows
      (H,H)  full projection matrix
    """
    if direction.dim() == 2 and direction.shape[0] == direction.shape[1]:
        z_clean = z @ direction.to(z.device).T
    elif direction.dim() == 2:
        basis = F.normalize(direction.to(z.device), dim=-1)
        z_clean = z - (z @ basis.T) @ basis
    else:
        direction = F.normalize(direction.to(z.device), dim=-1)
        z_clean = z - (z @ direction).unsqueeze(-1) * direction.unsqueeze(0)
    return F.normalize(z_clean, dim=-1)
