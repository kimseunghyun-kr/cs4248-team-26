"""
Combined debias_vl map discovery + CBDC text_iccv encoder training.

Phase A — debias_vl discovers confound "map" from word pairs:
  1. Encode (sentiment × topic) crossed prompts and pure topic prompts
  2. Compute debiasing projection P = P0 @ G^{-1}
  3. Extract confound directions via SVD(I - P)
  4. These directions become bias anchors for PGD

Phase B — CBDC text_iccv refines the map and trains the encoder:
  For each epoch:
    1. Bipolar PGD on target_text (class-conditioned prompts)
       using debias_vl-discovered directions as bias anchors
    2. S = z_adv_pos - z_adv_neg  (clean confound directions)
    3. Class-specific match_loss + ck_loss → update trainable tail

Architecture correspondence (RN50 ↔ BERT-derivative):
  CLIP:            frozen ResNet body → perturb → attnpool + c_proj (trainable tail)
  BERT-derivative: frozen layers 0–N  → perturb → layer N+1 (trainable tail)
  Both are single-attention-layer tails. This IS the RN50 approach.

Outputs:
  cache/debias_vl_P.pt             — (H, H) debiasing projection matrix
  cache/cbdc_directions.pt         — (K, H) PGD-refined confound directions
  cache/encoder_cbdc.pt            — fine-tuned tail layer checkpoint
  cache/z_tweet_{split}_cbdc.pt    — re-encoded embeddings

Run from project/ directory:
  python cbdc/refine.py [--n_epochs 100] [--lr 1e-5]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import CBDCConfig
from encoder import TransformerEncoder
from losses import l_bias_contrastive, l_semantic_preservation, l_ck
from cbdc.prompts import (
    get_prompt_bank,
    encode_all_prompts,
    flatten_prompt_groups,
    pool_prompt_group_embeddings,
)

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: debias_vl — discover confound map from word pairs
# ═══════════════════════════════════════════════════════════════════════════

def _get_A(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    """Asymmetric pairwise term (debias_vl.py::get_A)."""
    z_i = z_i[:, None]   # (H, 1)
    z_j = z_j[:, None]   # (H, 1)
    return z_i @ z_i.T + z_j @ z_j.T - z_i @ z_j.T - z_j @ z_i.T


def _get_M(embeddings: torch.Tensor, S: list) -> torch.Tensor:
    """Semantic preservation matrix (debias_vl.py::get_M)."""
    d = embeddings.shape[1]
    M = torch.zeros((d, d), device=embeddings.device)
    for s in S:
        M += _get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)


def _get_proj_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Orthogonal complement: P0 = I - V(V^T V)^{-1} V^T."""
    U, S, V = torch.svd(embeddings)
    basis = V
    proj_sup = basis @ torch.inverse(basis.T @ basis) @ basis.T
    return torch.eye(proj_sup.shape[0], device=embeddings.device) - proj_sup


def _instantiate_anchor_poles(
    spurious_cb: torch.Tensor,
    topics: list[str],
    svd_dirs: torch.Tensor,
    phrases_per_side: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Instantiate real pole anchors from mined topic prompts.

    For each top singular direction, select the highest- and lowest-scoring
    topic prompts and average them into two prompt-backed pole embeddings.
    """
    n_topics = len(topics)
    side_k = max(1, min(phrases_per_side, n_topics // 2))

    bias_anchors = []
    anti_anchors = []
    anchor_info = []

    for idx, direction in enumerate(F.normalize(svd_dirs, dim=-1)):
        scores = (spurious_cb @ direction).detach().cpu()
        pos_idx = torch.topk(scores, k=side_k).indices.tolist()
        neg_idx = torch.topk(-scores, k=side_k).indices.tolist()

        bias_anchor = F.normalize(spurious_cb[pos_idx].mean(0), dim=-1)
        anti_anchor = F.normalize(spurious_cb[neg_idx].mean(0), dim=-1)

        # Guard against degenerate poles by falling back to the SVD axis.
        pole_cos = F.cosine_similarity(
            bias_anchor.unsqueeze(0),
            anti_anchor.unsqueeze(0),
            dim=-1,
        ).item()
        if pole_cos > 0.98:
            bias_anchor = F.normalize(direction, dim=-1)
            anti_anchor = F.normalize(-direction, dim=-1)

        bias_anchors.append(bias_anchor)
        anti_anchors.append(anti_anchor)
        anchor_info.append(
            {
                "anchor_index": idx,
                "positive_topics": [topics[i] for i in pos_idx],
                "negative_topics": [topics[i] for i in neg_idx],
                "positive_scores": [float(scores[i]) for i in pos_idx],
                "negative_scores": [float(scores[i]) for i in neg_idx],
            }
        )

    return (
        torch.stack(bias_anchors, dim=0),
        torch.stack(anti_anchors, dim=0),
        anchor_info,
    )


def discover_confound_map(
    encoder: TransformerEncoder,
    cfg: CBDCConfig,
    prompt_bank: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """
    Phase A: Use debias_vl to discover confound directions from word pairs.

    Returns:
        P_debias:        (H, H)  debiasing projection matrix
        bias_anchors:    (K, H)  confound directions (bias anchors for PGD)
        anti_anchors:    (K, H)  opposing directions (negated)
        anchor_info:     prompt-backed pole descriptions for logging / inspection
    """
    print("\n=== Phase A: debias_vl confound map discovery ===")

    prompts = encode_all_prompts(encoder, prompt_bank)
    spurious_cb  = prompts["spurious_cb"]
    candidate_cb = prompts["candidate_cb"]
    S_pairs = prompt_bank["S_pairs"]
    B_pairs = prompt_bank["B_pairs"]
    topics = prompt_bank["topics"]

    print(f"  spurious_cb:  {tuple(spurious_cb.shape)}")
    print(f"  candidate_cb: {tuple(candidate_cb.shape)}")
    print(f"  S_pairs: {len(S_pairs)} | B_pairs: {len(B_pairs)}")
    source = "mined tweet phrases" if prompt_bank.get("using_mined_topics") else "static fallback topics"
    print(f"  topic source: {source} ({len(topics)} topics)")
    if prompt_bank.get("topic_metadata"):
        preview = ", ".join(topic["topic"] for topic in prompt_bank["topic_metadata"][:8])
        print(f"  mined preview: {preview}")
    if prompt_bank.get("mining_error"):
        print(f"  mining fallback reason: {prompt_bank['mining_error']}")

    # P0 = orthogonal complement of spurious subspace
    P0 = _get_proj_matrix(spurious_cb)

    # M = semantic preservation from S pairs
    M = _get_M(candidate_cb, S_pairs)

    # G = regularized semantic preservation
    H = candidate_cb.shape[1]
    G = cfg.lambda_reg * M + torch.eye(H, device=candidate_cb.device)

    # P_debias = P0 @ G^{-1}
    P_debias = P0 @ torch.inverse(G)

    # Extract confound directions via SVD(I - P_debias)
    confound_matrix = torch.eye(H, device=P_debias.device) - P_debias
    _, S_vals, Vh = torch.linalg.svd(confound_matrix, full_matrices=False)

    # Top-K directions
    K = cfg.n_bias_dirs
    svd_dirs = F.normalize(Vh[:K], dim=-1)
    bias_anchors, anti_anchors, anchor_info = _instantiate_anchor_poles(
        spurious_cb,
        topics,
        svd_dirs,
        phrases_per_side=cfg.pole_phrases_per_side,
    )

    print(f"  Top-{K} SVD singular values: {S_vals[:K].tolist()}")
    print(f"  bias_anchors: {tuple(bias_anchors.shape)}")

    # Diagnostic: check anchor quality
    for i in range(K):
        cls_cb = prompts["cls_cb"]  # (3, H)
        cos_cls = (bias_anchors[i] @ cls_cb.T).abs().max().item()
        print(f"    anchor {i}: max|cos(anchor, cls_em)|={cos_cls:.4f}  "
              f"(lower = less entangled with sentiment)")
        pos_topics = ", ".join(anchor_info[i]["positive_topics"])
        neg_topics = ", ".join(anchor_info[i]["negative_topics"])
        print(f"      pole A: {pos_topics}")
        print(f"      pole B: {neg_topics}")

    return P_debias, bias_anchors, anti_anchors, anchor_info


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: PGD inner loop — perturbs target_text concept prompts
# ═══════════════════════════════════════════════════════════════════════════

def _pgd_bipolar(
    h_target: torch.Tensor,         # (N_target, L, H) intermediate features of target_text
    attention_mask: torch.Tensor,    # (N_test, L)
    z_orig: torch.Tensor,           # (N_target, H) embeddings at delta=0
    bias_anchors: torch.Tensor,     # (K, H)
    anti_anchors: torch.Tensor,     # (K, H)
    keep_cb: torch.Tensor,          # (N_keep, H) for multi-axis L_s
    encoder: TransformerEncoder,
    cfg: CBDCConfig,
    sentiment_protos: torch.Tensor | None = None,  # (C, H) sentiment prototypes for orthogonal constraint
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bipolar PGD on concept prompt representations.

    Matches perturb_bafa_txt_multi_ablation_lb_ls:
      - Perturbs target_text intermediate features
      - L_B: cross-entropy over bias anchor pairs
      - L_s: multi-axis preservation via keep_cb
      - Sign-SGD + L∞ clamp

    Returns:
        z_adv_pos_all: (num_samples * N_target, H)  pushed toward bias_anchors
        z_adv_neg_all: (num_samples * N_target, H)  pushed toward anti_anchors
    """
    device = cfg.device
    N_target = h_target.shape[0]
    H = encoder.hidden_size

    bias_anchors = F.normalize(bias_anchors.to(device), dim=-1)
    anti_anchors = F.normalize(anti_anchors.to(device), dim=-1)
    keep_cb = F.normalize(keep_cb.to(device), dim=-1)

    # Precompute orthonormal sentiment basis for gradient projection
    S_basis = None
    if sentiment_protos is not None and cfg.sent_orthogonal_pgd:
        # Orthonormalize via QR to handle near-collinear prototypes
        S_basis = F.normalize(sentiment_protos.to(device), dim=-1)
        Q, R = torch.linalg.qr(S_basis.T)  # (H, C)
        S_basis = Q.T  # (C, H) orthonormal rows

    adv_pos_all = []
    adv_neg_all = []

    for restart in range(cfg.num_samples):
        # Random init for restarts > 0
        if restart == 0:
            init_pos = torch.zeros(N_target, H, device=device)
            init_neg = torch.zeros(N_target, H, device=device)
        else:
            init_pos = (torch.rand(N_target, H, device=device) * 2 - 1) * cfg.random_eps
            init_neg = (torch.rand(N_target, H, device=device) * 2 - 1) * cfg.random_eps

        # --- Positive pole: push toward bias_anchors (target=0) ---
        delta = init_pos.clone().requires_grad_(True)
        for _ in range(cfg.n_pgd_steps):
            if delta.grad is not None:
                delta.grad.zero_()
            z_pert = encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta)
            L_B = l_bias_contrastive(z_pert, bias_anchors, anti_anchors, push_toward_a=True)
            L_s = l_semantic_preservation(z_pert, z_orig, keep_cb)
            loss = L_B * (1.0 - cfg.keep_weight) - L_s * cfg.keep_weight
            loss.backward()
            with torch.no_grad():
                grad = delta.grad.data
                if S_basis is not None:
                    grad = grad - (grad @ S_basis.T) @ S_basis
                delta.data += cfg.step_lr * grad.sign()
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)
        with torch.no_grad():
            z_pos = encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta.detach())
        adv_pos_all.append(z_pos)

        # --- Negative pole: push toward anti_anchors (target=1) ---
        delta = init_neg.clone().requires_grad_(True)
        for _ in range(cfg.n_pgd_steps):
            if delta.grad is not None:
                delta.grad.zero_()
            z_pert = encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta)
            L_B = l_bias_contrastive(z_pert, bias_anchors, anti_anchors, push_toward_a=False)
            L_s = l_semantic_preservation(z_pert, z_orig, keep_cb)
            loss = L_B * (1.0 - cfg.keep_weight) - L_s * cfg.keep_weight
            loss.backward()
            with torch.no_grad():
                grad = delta.grad.data
                if S_basis is not None:
                    grad = grad - (grad @ S_basis.T) @ S_basis
                delta.data += cfg.step_lr * grad.sign()
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)
        with torch.no_grad():
            z_neg = encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta.detach())
        adv_neg_all.append(z_neg)

    return torch.cat(adv_pos_all, dim=0), torch.cat(adv_neg_all, dim=0)


def _encode_class_prototypes(
    encoder: TransformerEncoder,
    cls_ids: torch.Tensor,
    cls_mask: torch.Tensor,
    group_sizes: list[int],
) -> torch.Tensor:
    """Forward grouped class prompts through the trainable tail and pool them."""
    with torch.no_grad():
        h_cls = encoder.get_intermediate_features(cls_ids, cls_mask)

    cls_bank_em = encoder.encode_with_delta_from_hidden(
        h_cls,
        cls_mask,
        torch.zeros(len(cls_ids), encoder.hidden_size, device=cls_ids.device),
    )
    return pool_prompt_group_embeddings(cls_bank_em, group_sizes, normalize=True)


def _load_cached_split(split: str) -> dict:
    """Load cached tweet tensors from Phase 1."""
    path = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required cached split not found: {path}. Run data/embed.py first."
        )
    return torch.load(path, map_location="cpu")


def _select_balanced_indices(
    labels: torch.Tensor,
    max_per_class: int,
    seed: int = 42,
) -> torch.Tensor:
    """Choose a fixed balanced subset for validation-time selector scoring."""
    g = torch.Generator().manual_seed(seed)
    chosen = []
    for c in [0, 1, 2]:
        idx = torch.where(labels == c)[0]
        if len(idx) == 0:
            continue
        perm = idx[torch.randperm(len(idx), generator=g)]
        chosen.append(perm[:min(max_per_class, len(idx))])
    if not chosen:
        raise ValueError("Balanced selector sampling found no labeled examples.")
    return torch.cat(chosen, dim=0).sort().values


@torch.no_grad()
def _encode_cached_ids(
    encoder: TransformerEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Encode cached token IDs with the current encoder checkpoint."""
    all_z = []
    for i in range(0, len(input_ids), batch_size):
        z = encoder.encode_ids(
            input_ids[i:i+batch_size].to(encoder.device),
            attention_mask[i:i+batch_size].to(encoder.device),
        )
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


def _centroid_val_f1(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    """Validation macro F1 from a simple cosine nearest-centroid classifier."""
    centroids = []
    for c in [0, 1, 2]:
        mask = y_train == c
        if mask.sum() == 0:
            raise ValueError(f"Selector train subset has no samples for class {c}.")
        centroids.append(F.normalize(z_train[mask].mean(0), dim=-1))
    centroid_bank = torch.stack(centroids, dim=0)
    preds = (z_val @ centroid_bank.T).argmax(dim=-1).cpu().numpy()
    return f1_score(y_val.cpu().numpy(), preds, average="macro")


def _prepare_selector_data(cfg: CBDCConfig) -> dict:
    """Prepare a fixed balanced train subset plus full validation split."""
    train_data = _load_cached_split("train")
    val_data = _load_cached_split("val")

    train_labels = train_data["labels"].cpu()
    subset_idx = _select_balanced_indices(
        train_labels,
        max_per_class=cfg.selector_train_per_class,
    )

    return {
        "train_ids": train_data["input_ids"][subset_idx],
        "train_mask": train_data["attention_mask"][subset_idx],
        "train_labels": train_labels[subset_idx],
        "val_ids": val_data["input_ids"],
        "val_mask": val_data["attention_mask"],
        "val_labels": val_data["labels"].cpu(),
    }


@torch.no_grad()
def _selector_val_f1(
    encoder: TransformerEncoder,
    selector_data: dict,
    batch_size: int,
) -> float:
    """Encode selector splits and score them with cosine nearest-centroid F1."""
    z_train = _encode_cached_ids(
        encoder,
        selector_data["train_ids"],
        selector_data["train_mask"],
        batch_size=batch_size,
    )
    z_val = _encode_cached_ids(
        encoder,
        selector_data["val_ids"],
        selector_data["val_mask"],
        batch_size=batch_size,
    )
    return _centroid_val_f1(
        z_train,
        selector_data["train_labels"],
        z_val,
        selector_data["val_labels"],
    )


def _compute_direction_snapshot(
    encoder: TransformerEncoder,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    bias_anchors: torch.Tensor,
    anti_anchors: torch.Tensor,
    keep_cb: torch.Tensor,
    cfg: CBDCConfig,
    sentiment_protos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Re-run PGD with the current encoder to obtain an S snapshot."""
    with torch.no_grad():
        h_target = encoder.get_intermediate_features(target_ids, target_mask)
        z_orig = encoder.encode_with_delta_from_hidden(
            h_target,
            target_mask,
            torch.zeros(len(target_ids), encoder.hidden_size, device=cfg.device),
        )

    z_adv_pos, z_adv_neg = _pgd_bipolar(
        h_target,
        target_mask,
        z_orig,
        bias_anchors,
        anti_anchors,
        keep_cb,
        encoder,
        cfg,
        sentiment_protos=sentiment_protos,
    )
    return (z_adv_pos - z_adv_neg).detach().cpu()


# ═══════════════════════════════════════════════════════════════════════════
# Phase B+C: text_iccv training loop
# ═══════════════════════════════════════════════════════════════════════════

def text_iccv(
    encoder: TransformerEncoder,
    bias_anchors: torch.Tensor,     # (K, H) from debias_vl Phase A
    anti_anchors: torch.Tensor,     # (K, H)
    cfg: CBDCConfig,
    prompt_bank: dict,
) -> tuple[TransformerEncoder, torch.Tensor, torch.Tensor]:
    """
    CBDC text_iccv loop — iterative PGD + encoder training.

    Each epoch:
      1. PGD on target_text concept prompts
      2. S = z_adv_pos - z_adv_neg (confound direction per class-conditioned target)
      3. Class-specific match_loss + ck_loss → update trainable tail weights

    Returns:
        encoder:          best validation-selected TransformerEncoder
        final_directions: (K, H)  best-epoch PGD-refined directions
        final_cls_em:     (3, H)  best-epoch pooled sentiment prototypes
    """
    device = cfg.device
    _cls_text_groups = prompt_bank["cls_text_groups"]
    _cls_group_sizes = prompt_bank["cls_group_sizes"]
    _target_text = prompt_bank["target_text"]
    _keep_text = prompt_bank["keep_text"]

    n_cls = len(_cls_text_groups)
    n_target = len(_target_text)
    if n_target != n_cls:
        raise ValueError(
            f"RN50-style indexed match_loss expects len(target_text) == len(cls_text_groups); "
            f"got {n_target} vs {n_cls}"
        )

    print(f"\n=== Phase B+C: CBDC text_iccv training ===")
    print(f"  n_epochs={cfg.n_epochs} | lr={cfg.lr} | "
          f"PGD: ε={cfg.epsilon} steps={cfg.n_pgd_steps} restarts={cfg.num_samples}")

    # Tokenize concept prompts once
    cls_flat_text = flatten_prompt_groups(_cls_text_groups)
    cls_enc = encoder.tokenize(cls_flat_text)
    target_enc = encoder.tokenize(_target_text)
    cls_ids  = cls_enc["input_ids"].to(device)
    cls_mask = cls_enc["attention_mask"].to(device)
    target_ids = target_enc["input_ids"].to(device)
    target_mask = target_enc["attention_mask"].to(device)

    # keep_cb for L_s (semantic content we want to preserve)
    with torch.no_grad():
        keep_cb = encoder.encode_text(_keep_text).to(device)

    # Compute initial sentiment prototypes for orthogonal PGD constraint
    sentiment_protos = None
    if cfg.sent_orthogonal_pgd:
        with torch.no_grad():
            init_cls_em = _encode_class_prototypes(encoder, cls_ids, cls_mask, _cls_group_sizes)
        sentiment_protos = init_cls_em.detach().to(device)
        print(f"  Sentiment-orthogonal PGD: ON (projecting gradients ⊥ {sentiment_protos.shape[0]} prototypes)")

    # Set up trainable tail layer
    layer_tail = encoder._get_transformer_layers()[-1]
    for p in layer_tail.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(layer_tail.parameters(), lr=cfg.lr, weight_decay=1e-4)

    n_trainable = sum(p.numel() for p in layer_tail.parameters())
    print(f"  Trainable params (layer 11): {n_trainable:,}")
    print(f"  Selector: eval_every={cfg.eval_every} | "
          f"train_per_class={cfg.selector_train_per_class} | "
          f"batch_size={cfg.selector_batch_size}")

    selector_data = _prepare_selector_data(cfg)
    best_selector_f1 = float("-inf")
    best_epoch = None
    best_state = None

    for epoch in tqdm(range(cfg.n_epochs), desc="text_iccv"):
        # --- 1. Pre-compute target_text intermediate features (frozen body) ---
        with torch.no_grad():
            h_target = encoder.get_intermediate_features(target_ids, target_mask)
            z_orig = encoder.encode_with_delta_from_hidden(
                h_target,
                target_mask,
                torch.zeros(n_target, encoder.hidden_size, device=device),
            )

        # --- 2. Freeze layer 11 for PGD ---
        for p in layer_tail.parameters():
            p.requires_grad_(False)

        z_adv_pos, z_adv_neg = _pgd_bipolar(
            h_target,
            target_mask,
            z_orig,
            bias_anchors,
            anti_anchors,
            keep_cb,
            encoder, cfg,
            sentiment_protos=sentiment_protos,
        )

        # --- 3. Forward adversarial through layer 11 (frozen) to get S ---
        with torch.no_grad():
            S = z_adv_pos - z_adv_neg   # (num_samples * N_test, H)

        # --- 4. Unfreeze layer 11, compute cls_em WITH gradient ---
        for p in layer_tail.parameters():
            p.requires_grad_(True)

        cls_em = _encode_class_prototypes(encoder, cls_ids, cls_mask, _cls_group_sizes)

        # Update sentiment prototypes for next epoch's PGD constraint
        if sentiment_protos is not None:
            sentiment_protos = cls_em.detach().clone()

        # --- 5. Class-specific match_loss ---
        match_loss = torch.tensor(0.0, device=device)
        for c in range(n_target):
            match_loss += (S[c::n_target].to(device) @ cls_em[c:c+1].T).pow(2).mean()
        match_loss = match_loss * cfg.up_scale / n_target

        # --- 6. ck_loss ---
        ck_loss = l_ck(
            bias_anchors.to(device), anti_anchors.to(device),
            cls_em, scale=cfg.up_scale,
        )

        # --- 7. Backward + update ---
        total_loss = match_loss + ck_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        should_eval = (
            epoch == 0
            or (epoch + 1) % cfg.eval_every == 0
            or (epoch + 1) == cfg.n_epochs
        )

        selector_f1 = None
        if should_eval:
            selector_f1 = _selector_val_f1(
                encoder,
                selector_data,
                batch_size=cfg.selector_batch_size,
            )
            if selector_f1 > best_selector_f1:
                best_selector_f1 = selector_f1
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in layer_tail.state_dict().items()
                }

        if should_eval or (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                alignment = (S.to(device) @ cls_em.detach().T).pow(2).mean().item()
            selector_msg = ""
            if selector_f1 is not None:
                selector_msg = f" selector_f1={selector_f1:.4f}"
            print(f"  Epoch {epoch+1}/{cfg.n_epochs}: "
                  f"match={match_loss.item():.4f} ck={ck_loss.item():.4f} "
                  f"total={total_loss.item():.4f} alignment={alignment:.6f}"
                  f"{selector_msg}")

    if best_state is not None:
        layer_tail.load_state_dict(best_state)
        print(f"\n  Restored best CBDC epoch: {best_epoch}/{cfg.n_epochs} "
              f"(selector_f1={best_selector_f1:.4f})")

    # Freeze after training
    for p in layer_tail.parameters():
        p.requires_grad_(False)
    encoder.backbone.eval()

    # Recompute PGD snapshot using the best encoder checkpoint.
    best_S = _compute_direction_snapshot(
        encoder,
        target_ids,
        target_mask,
        bias_anchors,
        anti_anchors,
        keep_cb,
        cfg,
        sentiment_protos=sentiment_protos,
    )

    # Extract final directions from the best-S snapshot via SVD
    S_centered = best_S - best_S.mean(0)
    _, _, Vh = torch.linalg.svd(S_centered, full_matrices=False)
    final_directions = Vh[:cfg.n_bias_dirs]  # (K, H) orthonormal

    print(f"\n  Final directions shape: {tuple(final_directions.shape)}")
    gram = final_directions @ final_directions.T
    ortho_err = (gram - torch.eye(cfg.n_bias_dirs)).abs().max().item()
    print(f"  Orthonormality error: {ortho_err:.2e}")

    # Encode class prompt bank with the final encoder and pool to 3 prototypes.
    with torch.no_grad():
        cls_bank_final = encoder.encode_text(cls_flat_text).cpu()
        final_cls_em = pool_prompt_group_embeddings(
            cls_bank_final,
            _cls_group_sizes,
            normalize=True,
        )

    return encoder, final_directions, final_cls_em


# ═══════════════════════════════════════════════════════════════════════════
# Re-encode all splits with fine-tuned encoder
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def reencode_splits(encoder: TransformerEncoder, device: str, suffix: str = "cbdc"):
    """Re-encode train/val/test with the fine-tuned encoder."""
    for split in ["train", "val", "test"]:
        in_path = os.path.join(CACHE_DIR, f"z_tweet_{split}.pt")
        if not os.path.exists(in_path):
            print(f"  [skip] Missing {in_path}")
            continue

        data = torch.load(in_path, map_location="cpu")
        ids = data["input_ids"]
        mask = data["attention_mask"]
        labels = data.get("labels")

        batch_size = 128
        all_z = []
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size].to(device)
            batch_mask = mask[i:i+batch_size].to(device)
            z = encoder.encode_ids(batch_ids, batch_mask)
            all_z.append(z.cpu())

        z_new = torch.cat(all_z, dim=0)

        out_data = {"embeddings": z_new}
        if labels is not None:
            out_data["labels"] = labels
        out_data["input_ids"] = ids
        out_data["attention_mask"] = mask

        out_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_{suffix}.pt")
        torch.save(out_data, out_path)
        print(f"  {split}: {tuple(z_new.shape)} -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Combined debias_vl + CBDC text_iccv training"
    )
    parser.add_argument("--n_epochs",    type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--epsilon",     type=float, default=1.0)
    parser.add_argument("--n_pgd_steps", type=int,   default=20)
    parser.add_argument("--step_lr",     type=float, default=0.0037)
    parser.add_argument("--keep_weight", type=float, default=0.92)
    parser.add_argument("--num_samples", type=int,   default=10)
    parser.add_argument("--random_eps",  type=float, default=0.22)
    parser.add_argument("--n_bias_dirs", type=int,   default=4)
    parser.add_argument("--lambda_reg",  type=float, default=1000.0)
    parser.add_argument("--up_scale",    type=float, default=100.0)
    parser.add_argument("--eval_every",  type=int,   default=10)
    parser.add_argument("--selector_train_per_class", type=int, default=512)
    parser.add_argument("--selector_batch_size", type=int, default=128)
    parser.add_argument("--use_static_topics", action="store_true",
                        help="Disable tweet phrase mining and use the static debias_vl topic list.")
    parser.add_argument("--mine_max_topics", type=int, default=32)
    parser.add_argument("--mine_min_doc_freq", type=int, default=20)
    parser.add_argument("--mine_max_doc_freq_ratio", type=float, default=0.20)
    parser.add_argument("--pole_phrases_per_side", type=int, default=4)
    parser.add_argument("--refresh_mined_topics", action="store_true",
                        help="Ignore cached mined_topics.json and mine the topic bank again.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Build config -------------------------------------------------------
    cfg = CBDCConfig(
        epsilon=args.epsilon,
        n_pgd_steps=args.n_pgd_steps,
        step_lr=args.step_lr,
        keep_weight=args.keep_weight,
        num_samples=args.num_samples,
        random_eps=args.random_eps,
        n_epochs=args.n_epochs,
        lr=args.lr,
        up_scale=args.up_scale,
        eval_every=args.eval_every,
        selector_train_per_class=args.selector_train_per_class,
        selector_batch_size=args.selector_batch_size,
        use_mined_topics=not args.use_static_topics,
        mine_max_topics=args.mine_max_topics,
        mine_min_doc_freq=args.mine_min_doc_freq,
        mine_max_doc_freq_ratio=args.mine_max_doc_freq_ratio,
        pole_phrases_per_side=args.pole_phrases_per_side,
        n_bias_dirs=args.n_bias_dirs,
        lambda_reg=args.lambda_reg,
        device=device,
    )

    # ---- Load encoder (with optional custom tokenizer) -----------------------
    model_name = os.environ.get("MODEL_NAME", "bert-base-uncased")
    tokenizer = None
    tokenizer_name = os.environ.get("TOKENIZER_NAME")
    if tokenizer_name:
        from transformers import AutoTokenizer
        print(f"Loading custom tokenizer from '{tokenizer_name}' ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoder = TransformerEncoder(model_name=model_name, device=device, tokenizer=tokenizer)

    # ---- Prompt bank --------------------------------------------------------
    text_unit = os.environ.get("TEXT_UNIT", cfg.text_unit)
    prompt_bank = get_prompt_bank(
        tokenizer=encoder.tokenizer,
        cache_dir=CACHE_DIR,
        use_mined_topics=cfg.use_mined_topics,
        max_topics=cfg.mine_max_topics,
        min_doc_freq=cfg.mine_min_doc_freq,
        max_doc_freq_ratio=cfg.mine_max_doc_freq_ratio,
        force_refresh=args.refresh_mined_topics,
        text_unit=text_unit,
    )
    prompt_bank_path = os.path.join(CACHE_DIR, "prompt_bank.json")
    with open(prompt_bank_path, "w") as f:
        json.dump(
            {
                "using_mined_topics": prompt_bank.get("using_mined_topics", False),
                "topics": prompt_bank["topics"],
                "topic_metadata": prompt_bank.get("topic_metadata", []),
                "mining_error": prompt_bank.get("mining_error"),
            },
            f,
            indent=2,
        )
    print(f"  prompt bank -> {prompt_bank_path}")

    # ---- Phase A: debias_vl map discovery -----------------------------------
    P_debias, bias_anchors, anti_anchors, anchor_info = discover_confound_map(
        encoder,
        cfg,
        prompt_bank,
    )

    # Save debias_vl projection matrix
    p_path = os.path.join(CACHE_DIR, "debias_vl_P.pt")
    torch.save(P_debias.cpu(), p_path)
    print(f"  debias_vl projection -> {p_path}")

    anchor_path = os.path.join(CACHE_DIR, "anchor_poles.json")
    with open(anchor_path, "w") as f:
        json.dump(anchor_info, f, indent=2)
    print(f"  anchor poles -> {anchor_path}")

    # ---- Phase B+C: CBDC text_iccv training ---------------------------------
    encoder, final_directions, final_cls_prototypes = text_iccv(encoder, bias_anchors, anti_anchors, cfg, prompt_bank)

    # Save outputs
    dir_path = os.path.join(CACHE_DIR, "cbdc_directions.pt")
    torch.save(final_directions, dir_path)
    print(f"  CBDC directions ({tuple(final_directions.shape)}) -> {dir_path}")

    sent_path = os.path.join(CACHE_DIR, "sentiment_prototypes.pt")
    torch.save(final_cls_prototypes, sent_path)
    print(f"  Sentiment prototypes ({tuple(final_cls_prototypes.shape)}) -> {sent_path}")

    ckpt_path = os.path.join(CACHE_DIR, "encoder_cbdc.pt")
    torch.save(encoder._get_transformer_layers()[-1].state_dict(), ckpt_path)
    print(f"  Encoder checkpoint -> {ckpt_path}")

    # ---- Re-encode with fine-tuned encoder ----------------------------------
    print("\nRe-encoding all splits with fine-tuned encoder ...")
    reencode_splits(encoder, device, suffix="cbdc")

    # ---- Apply residual projection on CBDC-encoded embeddings ---------------
    print("\nApplying residual CBDC projection on fine-tuned embeddings ...")
    from pipeline.clean import project_out
    for split in ["train", "val", "test"]:
        in_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_cbdc.pt")
        if not os.path.exists(in_path):
            continue
        data = torch.load(in_path, map_location="cpu")
        z = data["embeddings"]
        z_clean = project_out(z, final_directions)
        out_data = {"embeddings": z_clean}
        if "labels" in data:
            out_data["labels"] = data["labels"]
        out_path = os.path.join(CACHE_DIR, f"z_tweet_{split}_clean_cbdc_proj.pt")
        torch.save(out_data, out_path)
        print(f"  {split}: CBDC+proj -> {out_path}")

    print("\nCBDC training complete.")


if __name__ == "__main__":
    main()
