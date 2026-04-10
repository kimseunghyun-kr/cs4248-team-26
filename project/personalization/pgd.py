from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from encoder import TransformerEncoder
from losses import l_bias_contrastive, l_semantic_preservation
from .prompts import PersonalizationPromptBank


@dataclass
class DiscoveryArtifacts:
    directions: torch.Tensor
    bias_anchors: torch.Tensor
    anti_anchors: torch.Tensor
    target_embeddings: torch.Tensor
    target_prompts: list[str]
    axis_names: list[str]
    singular_values: list[float]


def _build_anchor_bank(
    encoder: TransformerEncoder,
    prompt_bank: PersonalizationPromptBank,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    pos_bank = []
    neg_bank = []
    axis_names = []
    for axis in prompt_bank.nuisance_axes:
        pos = encoder.encode_text(axis.positive_prompts).mean(0)
        neg = encoder.encode_text(axis.negative_prompts).mean(0)
        pos_bank.append(F.normalize(pos.float(), dim=-1))
        neg_bank.append(F.normalize(neg.float(), dim=-1))
        axis_names.append(axis.name)

    return torch.stack(pos_bank, dim=0), torch.stack(neg_bank, dim=0), axis_names


def _encode_identity_basis(
    encoder: TransformerEncoder,
    prompt_bank: PersonalizationPromptBank,
) -> torch.Tensor:
    return encoder.encode_text(prompt_bank.identity_prompts)


def _pgd_bipolar(
    h_target: torch.Tensor,
    attention_mask: torch.Tensor,
    z_orig: torch.Tensor,
    bias_anchors: torch.Tensor,
    anti_anchors: torch.Tensor,
    keep_cb: torch.Tensor,
    identity_basis: torch.Tensor,
    encoder: TransformerEncoder,
    epsilon: float,
    step_lr: float,
    n_steps: int,
    num_restarts: int,
    random_eps: float,
    keep_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = h_target.device
    n_target = h_target.shape[0]
    hidden_size = encoder.hidden_size

    bias_anchors = F.normalize(bias_anchors.to(device), dim=-1)
    anti_anchors = F.normalize(anti_anchors.to(device), dim=-1)
    keep_cb = F.normalize(keep_cb.to(device), dim=-1)

    identity_basis = F.normalize(identity_basis.to(device), dim=-1)
    basis_q, _ = torch.linalg.qr(identity_basis.T)
    identity_basis = basis_q.T

    adv_pos_all = []
    adv_neg_all = []

    for restart in range(num_restarts):
        if restart == 0:
            init_pos = torch.zeros(n_target, hidden_size, device=device)
            init_neg = torch.zeros(n_target, hidden_size, device=device)
        else:
            init_pos = (torch.rand(n_target, hidden_size, device=device) * 2 - 1) * random_eps
            init_neg = (torch.rand(n_target, hidden_size, device=device) * 2 - 1) * random_eps

        for seed_delta, push_toward_a, bank in [
            (init_pos, True, adv_pos_all),
            (init_neg, False, adv_neg_all),
        ]:
            delta = seed_delta.clone().requires_grad_(True)
            for _ in range(n_steps):
                if delta.grad is not None:
                    delta.grad.zero_()

                z_pert = encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta)
                bias_loss = l_bias_contrastive(
                    z_pert,
                    bias_anchors,
                    anti_anchors,
                    push_toward_a=push_toward_a,
                )
                keep_loss = l_semantic_preservation(z_pert, z_orig, keep_cb)
                loss = bias_loss * (1.0 - keep_weight) - keep_loss * keep_weight
                loss.backward()

                with torch.no_grad():
                    grad = delta.grad.data
                    grad = grad - (grad @ identity_basis.T) @ identity_basis
                    delta.data += step_lr * grad.sign()
                    delta.data.clamp_(-epsilon, epsilon)

            with torch.no_grad():
                bank.append(
                    encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta.detach())
                )

    return torch.cat(adv_pos_all, dim=0), torch.cat(adv_neg_all, dim=0)


def discover_nuisance_directions(
    encoder: TransformerEncoder,
    prompt_bank: PersonalizationPromptBank,
    epsilon: float,
    step_lr: float,
    n_steps: int,
    num_restarts: int,
    random_eps: float,
    keep_weight: float,
    n_bias_dirs: int,
) -> DiscoveryArtifacts:
    device = encoder.device
    bias_anchors, anti_anchors, axis_names = _build_anchor_bank(encoder, prompt_bank)
    keep_cb = encoder.encode_text(prompt_bank.keep_prompts).to(device)
    identity_basis = _encode_identity_basis(encoder, prompt_bank).to(device)

    tokenized = encoder.tokenize(prompt_bank.target_prompts)
    target_ids = tokenized["input_ids"].to(device)
    target_mask = tokenized["attention_mask"].to(device)
    h_target = encoder.get_intermediate_features(target_ids, target_mask)

    with torch.no_grad():
        z_orig = encoder.encode_with_delta_from_hidden(
            h_target,
            target_mask,
            torch.zeros(len(target_ids), encoder.hidden_size, device=device),
        )

    z_adv_pos, z_adv_neg = _pgd_bipolar(
        h_target=h_target,
        attention_mask=target_mask,
        z_orig=z_orig,
        bias_anchors=bias_anchors,
        anti_anchors=anti_anchors,
        keep_cb=keep_cb,
        identity_basis=identity_basis,
        encoder=encoder,
        epsilon=epsilon,
        step_lr=step_lr,
        n_steps=n_steps,
        num_restarts=num_restarts,
        random_eps=random_eps,
        keep_weight=keep_weight,
    )

    delta_bank = (z_adv_pos - z_adv_neg).float()
    centered = delta_bank - delta_bank.mean(0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    k = max(1, min(n_bias_dirs, vh.shape[0]))

    return DiscoveryArtifacts(
        directions=F.normalize(vh[:k], dim=-1).cpu(),
        bias_anchors=bias_anchors.cpu(),
        anti_anchors=anti_anchors.cpu(),
        target_embeddings=z_orig.cpu(),
        target_prompts=list(prompt_bank.target_prompts),
        axis_names=axis_names,
        singular_values=[float(value) for value in singular_values[:k].tolist()],
    )
