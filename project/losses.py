"""
Loss functions for CBDC direction discovery and encoder training.

L_B   — contrastive bias alignment (CLIP-style multi-anchor cross-entropy)
L_s   — multi-axis semantic preservation (matches original exactly)
L_ck  — cross-knowledge loss (penalizes bias-class alignment)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# L_B: contrastive bias alignment (CLIP-style)
# ---------------------------------------------------------------------------
def l_bias_contrastive(
    z_pert: torch.Tensor,          # (B, H)  normalized perturbed embedding
    bias_anchors: torch.Tensor,    # (K, H)  pole-A anchor directions
    anti_anchors: torch.Tensor,    # (K, H)  pole-B anchor directions
    push_toward_a: bool,
    temperature: float = 100.0,
) -> torch.Tensor:
    """
    CLIP-style pairwise cross-entropy over K anchor pairs.

    logits_i = [z · bias_i,  z · anti_i]   for i = 0..K-1
    L_B = cross_entropy(temperature * logits, target_class)

    push_toward_a=True  → target=0 (maximize z · bias_i)
    push_toward_a=False → target=1 (maximize z · anti_i)
    """
    B = z_pert.shape[0]
    K = bias_anchors.shape[0]
    device = z_pert.device

    bias_anchors = F.normalize(bias_anchors.to(device), dim=-1)
    anti_anchors = F.normalize(anti_anchors.to(device), dim=-1)

    sim_a = z_pert @ bias_anchors.T    # (B, K)
    sim_b = z_pert @ anti_anchors.T    # (B, K)

    logits = torch.stack([sim_a, sim_b], dim=-1).view(B * K, 2) * temperature

    target_class = 0 if push_toward_a else 1
    labels = torch.full((B * K,), target_class, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# L_s: multi-axis semantic preservation (matches original exactly)
# ---------------------------------------------------------------------------
def l_semantic_preservation(
    z_pert: torch.Tensor,      # (B, H)  normalized
    z_orig: torch.Tensor,      # (B, H)  normalized, no grad needed
    keep_axes: torch.Tensor,   # (N_keep, H)  multiple neutral concept embeddings
) -> torch.Tensor:
    """
    Original formula (simple_pgd.py line 1391):
        keep_loss = 100 * ((adv_feat - ori) @ keep.T).pow(2).mean()

    Penalizes change in projection onto ALL neutral concept directions,
    not just a single axis. keep_axes = test_cb (multiple neutral embeddings).
    """
    keep_axes = F.normalize(keep_axes.to(z_pert.device), dim=-1)   # (N_keep, H)
    delta = z_pert - z_orig.detach()                                # (B, H)
    proj = delta @ keep_axes.T                                      # (B, N_keep)
    return 100.0 * proj.pow(2).mean()


# ---------------------------------------------------------------------------
# L_ck: cross-knowledge loss
# ---------------------------------------------------------------------------
def l_ck(
    bias_a: torch.Tensor,     # (N, H)  pole-A embeddings (e.g. topic-confound directions)
    bias_b: torch.Tensor,     # (N, H)  pole-B embeddings (opposing directions)
    cls_em: torch.Tensor,     # (C, H)  class concept embeddings
    scale: float = 100.0,
) -> torch.Tensor:
    """
    Original (base.py line 225):
        ck_loss = ((bias_cb[:4] - bias_cb[4:]) @ cls_em.T).pow(2).mean() * up_

    Penalizes alignment between confound-pair differences and class embeddings.
    If (bias_a - bias_b) is orthogonal to cls_em, ck_loss ≈ 0.
    """
    diff = bias_a - bias_b                       # (N, H)
    return (diff @ cls_em.T).pow(2).mean() * scale
