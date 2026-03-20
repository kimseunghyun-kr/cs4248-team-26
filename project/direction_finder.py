"""
PGD-based direction discovery in FinBERT latent space.

For each concept axis we run PGD on a batch of anchor texts:
  δ* = argmin_δ  L_B(δ) - λ * L_s(δ)   s.t. ||δ||_∞ ≤ ε

The direction is derived from the converged δ*:
  monopolar: normalize(ϕ(z+δ*) - ϕ(z))
  bipolar:   normalize(ϕ(z+δ*) - ϕ(z-δ*))
"""

import torch
import torch.nn.functional as F
from typing import Optional

from encoder import FinBERTEncoder
from losses import l_bias_monopolar, l_bias_bipolar, l_semantic_preservation
from config import PGDConfig


class PGDDirectionFinder:
    def __init__(self, encoder: FinBERTEncoder, cfg: PGDConfig):
        self.encoder = encoder
        self.cfg = cfg
        self.device = cfg.device

    # ------------------------------------------------------------------
    # Internal PGD loop shared by both modes
    # ------------------------------------------------------------------
    def _run_pgd(
        self,
        input_ids: torch.Tensor,       # B × L
        attention_mask: torch.Tensor,  # B × L
        v_neutral: torch.Tensor,       # H
        compute_loss_fn,               # callable(z_perturbed, z_neg_or_none) → scalar
        bipolar: bool = False,
    ) -> torch.Tensor:
        """
        Returns converged δ* of shape B × H.
        Uses Adam (not sign-SGD) as specified in the brief.
        """
        B = input_ids.size(0)
        H = self.encoder.hidden_size
        eps = self.cfg.epsilon
        lam = self.cfg.lambda_s

        # Get clean embeddings (no grad, used for L_s reference)
        with torch.no_grad():
            z_orig = self.encoder.encode_ids(input_ids, attention_mask)  # B×H

        # Initialize δ as small random noise clipped to ε-ball
        delta = torch.zeros(B, H, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.cfg.step_lr)

        for step in range(self.cfg.n_steps):
            optimizer.zero_grad()

            # Forward: encode with +δ
            z_pos = self.encoder.encode_with_delta(input_ids, attention_mask, delta)

            if bipolar:
                # Also encode with -δ
                z_neg = self.encoder.encode_with_delta(
                    input_ids, attention_mask, -delta
                )
                L_b = compute_loss_fn(z_pos, z_neg)
            else:
                L_b = compute_loss_fn(z_pos)

            # Semantic preservation on the +δ branch
            L_s = l_semantic_preservation(z_pos, z_orig, v_neutral)

            # L_b is already negated (= -cosine_sim), so we minimize:
            #   L_b + λ * L_s  ≡  maximize cosine_sim  AND  minimize semantic drift
            # (Brief eq.10 uses gradient ascent on L_B - λ*L_s where L_B=+cosine_sim;
            #  gradient descent on its negation gives: -cosine_sim + λ*L_s)
            loss = L_b + lam * L_s
            loss.backward()
            optimizer.step()

            # Project δ back into ε-ball (ℓ∞ constraint)
            with torch.no_grad():
                delta.data = delta.data.clamp(-eps, eps)

        return delta.detach()

    # ------------------------------------------------------------------
    # Public: find a single monopolar direction from one batch of anchors
    # ------------------------------------------------------------------
    def find_monopolar(
        self,
        anchor_texts: list[str],
        target_text: str,
        neutral_text: str,
    ) -> torch.Tensor:
        """
        Returns a unit direction vector (H,).
        """
        # Encode concept anchors
        v_target = self.encoder.encode_text([target_text]).squeeze(0)    # H
        v_neutral = self.encoder.encode_text([neutral_text]).squeeze(0)  # H

        enc = self.encoder.tokenize(anchor_texts)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        loss_fn = lambda z_pos: l_bias_monopolar(z_pos, v_target)

        delta_star = self._run_pgd(
            input_ids, attention_mask, v_neutral, loss_fn, bipolar=False
        )  # B × H

        # Direction = mean of (ϕ(z+δ*) - ϕ(z)) over anchors
        with torch.no_grad():
            z_orig = self.encoder.encode_ids(input_ids, attention_mask)
            z_pert = self.encoder.encode_with_delta(
                input_ids, attention_mask, delta_star
            )
            direction = (z_pert - z_orig).mean(dim=0)  # H

        return F.normalize(direction, dim=-1)

    # ------------------------------------------------------------------
    # Public: find a single bipolar direction from one batch of anchors
    # ------------------------------------------------------------------
    def find_bipolar(
        self,
        anchor_texts: list[str],
        pos_text: str,
        neg_text: str,
        neutral_text: str,
    ) -> torch.Tensor:
        """
        Returns a unit direction vector (H,).
        """
        v_pos = self.encoder.encode_text([pos_text]).squeeze(0)
        v_neg = self.encoder.encode_text([neg_text]).squeeze(0)
        v_neutral = self.encoder.encode_text([neutral_text]).squeeze(0)

        enc = self.encoder.tokenize(anchor_texts)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        loss_fn = lambda z_pos, z_neg: l_bias_bipolar(z_pos, z_neg, v_pos, v_neg)

        delta_star = self._run_pgd(
            input_ids, attention_mask, v_neutral, loss_fn, bipolar=True
        )  # B × H

        # Direction = mean of (ϕ(z+δ*) - ϕ(z-δ*)) over anchors
        with torch.no_grad():
            z_p = self.encoder.encode_with_delta(input_ids, attention_mask, delta_star)
            z_n = self.encoder.encode_with_delta(input_ids, attention_mask, -delta_star)
            direction = (z_p - z_n).mean(dim=0)  # H

        return F.normalize(direction, dim=-1)

    # ------------------------------------------------------------------
    # Collect multiple directions for one axis (run PGD on sub-batches)
    # ------------------------------------------------------------------
    def collect_directions(
        self,
        mode: str,                  # "monopolar" or "bipolar"
        anchor_texts: list[str],
        n_directions: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Runs PGD n_directions times on shuffled sub-batches of anchors.
        Returns tensor of shape (n_directions, H).
        """
        directions = []
        batch_size = min(16, len(anchor_texts))

        for i in range(n_directions):
            # Randomly sample a sub-batch of anchors
            indices = torch.randperm(len(anchor_texts))[:batch_size].tolist()
            batch = [anchor_texts[j] for j in indices]

            if mode == "monopolar":
                d = self.find_monopolar(batch, **kwargs)
            else:
                d = self.find_bipolar(batch, **kwargs)

            directions.append(d)

        return torch.stack(directions, dim=0)  # N × H
