"""
Loss functions for PGD direction discovery.

L_B   — bias/concept alignment loss (monopolar or bipolar)
L_s   — semantic preservation loss
"""

import torch
import torch.nn.functional as F
from typing import Optional


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch cosine similarity: (B,H) x (H,) or (B,H) x (B,H) → (B,)"""
    return F.cosine_similarity(a, b.expand_as(a), dim=-1)


# ---------------------------------------------------------------------------
# L_B monopolar
# ---------------------------------------------------------------------------
def l_bias_monopolar(
    z_perturbed: torch.Tensor,   # B × H  (normalized)
    v_target: torch.Tensor,      # H      (normalized concept anchor)
) -> torch.Tensor:
    """
    Maximize alignment of perturbed CLS with target concept.
    Returns scalar loss (to be minimized → negative cosine).
    """
    v_target = F.normalize(v_target, dim=-1)
    sim = cosine_sim(z_perturbed, v_target)   # B
    return -sim.mean()


# ---------------------------------------------------------------------------
# L_B bipolar
# ---------------------------------------------------------------------------
def l_bias_bipolar(
    z_pos: torch.Tensor,    # B × H  (z + δ, normalized)
    z_neg: torch.Tensor,    # B × H  (z - δ, normalized)
    v_positive: torch.Tensor,  # H
    v_negative: torch.Tensor,  # H
) -> torch.Tensor:
    """
    Push (z+δ) toward positive concept AND (z-δ) toward negative concept.
    Returns scalar loss.
    """
    v_positive = F.normalize(v_positive, dim=-1)
    v_negative = F.normalize(v_negative, dim=-1)
    sim_pos = cosine_sim(z_pos, v_positive).mean()
    sim_neg = cosine_sim(z_neg, v_negative).mean()
    return -(sim_pos + sim_neg) / 2.0


# ---------------------------------------------------------------------------
# L_s  semantic preservation
# ---------------------------------------------------------------------------
def l_semantic_preservation(
    z_perturbed: torch.Tensor,   # B × H  (normalized)
    z_original: torch.Tensor,    # B × H  (normalized, no grad needed)
    v_neutral: torch.Tensor,     # H      (normalized neutral anchor)
) -> torch.Tensor:
    """
    Penalize change in projection onto neutral semantic axis.

    L_s = || v_C · (z_perturbed - z_original)^T ||^2_F
    averaged over batch.
    """
    v_neutral = F.normalize(v_neutral, dim=-1)          # H
    delta_z = z_perturbed - z_original.detach()         # B × H
    # projection of each delta_z onto v_neutral: scalar per sample
    proj = (delta_z * v_neutral.unsqueeze(0)).sum(dim=-1)  # B
    return (proj ** 2).mean()
