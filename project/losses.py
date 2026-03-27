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
# L_B contrastive (CLIP-style multi-anchor cross-entropy)
# ---------------------------------------------------------------------------
def l_bias_contrastive(
    z_pert: torch.Tensor,          # (B, H)  normalized perturbed embedding
    style_anchors: torch.Tensor,   # (K, H)  normalized tweet-style anchor directions
    anti_anchors: torch.Tensor,    # (K, H)  normalized formal-style anchor directions
    push_toward_style: bool,
    temperature: float = 100.0,
) -> torch.Tensor:
    """
    CLIP-style pairwise cross-entropy L_B with a multi-anchor style set.

    NLP analog of perturb_bafa_txt_multi_ablation_lb_ls (simple_pgd.py):
      logits_i = [z · style_i,  z · anti_i]  for i = 0..K-1
      L_B = cross_entropy(temperature * logits, target_class)

    Positive pole (push_toward_style=True):
      target = 0  →  maximize z · style_i  (confidence in style class decays naturally)
    Negative pole (push_toward_style=False):
      target = 1  →  maximize z · anti_i   (push toward formal/anti-style)

    Advantages over cosine L_B:
      * Gradient decays to zero when prediction is confident (no saturation waste)
      * Contrastive: simultaneously pulls toward style AND pushes away from anti-style
      * Multi-anchor: K directions, not a single fixed v_style

    Shape: logits (B*K, 2), labels (B*K,) → scalar loss.
    """
    B = z_pert.shape[0]
    K = style_anchors.shape[0]
    device = z_pert.device

    style_anchors = F.normalize(style_anchors.to(device), dim=-1)  # (K, H)
    anti_anchors  = F.normalize(anti_anchors.to(device),  dim=-1)  # (K, H)

    # (B, K) pairwise dot products
    sim_style = z_pert @ style_anchors.T   # (B, K)
    sim_anti  = z_pert @ anti_anchors.T    # (B, K)

    # Stack to (B, K, 2) → (B*K, 2), scale for softmax sharpness
    logits = torch.stack([sim_style, sim_anti], dim=-1).view(B * K, 2) * temperature

    target_class = 0 if push_toward_style else 1
    labels = torch.full((B * K,), target_class, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


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
