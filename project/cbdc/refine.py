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
    3. Class-specific match_loss + ck_loss → update layer 11

Architecture correspondence (RN50 ↔ FinBERT):
  CLIP:    frozen ResNet body → perturb → attnpool + c_proj (trainable tail)
  FinBERT: frozen layers 0–10 → perturb → layer 11 (trainable tail)
  Both are single-attention-layer tails. This IS the RN50 approach.

Outputs:
  cache/debias_vl_P.pt             — (H, H) debiasing projection matrix
  cache/cbdc_directions.pt         — (K, H) PGD-refined confound directions
  cache/encoder_cbdc.pt            — fine-tuned layer 11 checkpoint
  cache/z_tweet_{split}_cbdc.pt    — re-encoded embeddings

Run from project/ directory:
  python cbdc/refine.py [--n_epochs 100] [--lr 1e-5]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CBDCConfig
from encoder import FinBERTEncoder
from losses import l_bias_contrastive, l_semantic_preservation, l_ck
from cbdc.prompts import (
    cls_text,
    cls_text_groups,
    cls_group_sizes,
    target_text,
    keep_text,
    S_pairs, B_pairs,
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


def discover_confound_map(
    encoder: FinBERTEncoder,
    cfg: CBDCConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase A: Use debias_vl to discover confound directions from word pairs.

    Returns:
        P_debias:        (H, H)  debiasing projection matrix
        bias_anchors:    (K, H)  confound directions (bias anchors for PGD)
        anti_anchors:    (K, H)  opposing directions (negated)
    """
    print("\n=== Phase A: debias_vl confound map discovery ===")

    prompts = encode_all_prompts(encoder)
    spurious_cb  = prompts["spurious_cb"]
    candidate_cb = prompts["candidate_cb"]

    print(f"  spurious_cb:  {tuple(spurious_cb.shape)}")
    print(f"  candidate_cb: {tuple(candidate_cb.shape)}")
    print(f"  S_pairs: {len(S_pairs)} | B_pairs: {len(B_pairs)}")

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
    bias_anchors = F.normalize(Vh[:K], dim=-1)      # (K, H)
    anti_anchors = F.normalize(-Vh[:K], dim=-1)      # (K, H) opposing

    print(f"  Top-{K} SVD singular values: {S_vals[:K].tolist()}")
    print(f"  bias_anchors: {tuple(bias_anchors.shape)}")

    # Diagnostic: check anchor quality
    for i in range(K):
        cls_cb = prompts["cls_cb"]  # (3, H)
        cos_cls = (bias_anchors[i] @ cls_cb.T).abs().max().item()
        print(f"    anchor {i}: max|cos(anchor, cls_em)|={cos_cls:.4f}  "
              f"(lower = less entangled with sentiment)")

    return P_debias, bias_anchors, anti_anchors


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
    encoder: FinBERTEncoder,
    cfg: CBDCConfig,
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
                # Match RN50 text_iccv / simple_pgd: ascend on the PGD objective.
                delta.data += cfg.step_lr * delta.grad.sign()
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
                delta.data += cfg.step_lr * delta.grad.sign()
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)
        with torch.no_grad():
            z_neg = encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta.detach())
        adv_neg_all.append(z_neg)

    return torch.cat(adv_pos_all, dim=0), torch.cat(adv_neg_all, dim=0)


def _encode_class_prototypes(
    encoder: FinBERTEncoder,
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


# ═══════════════════════════════════════════════════════════════════════════
# Phase B+C: text_iccv training loop
# ═══════════════════════════════════════════════════════════════════════════

def text_iccv(
    encoder: FinBERTEncoder,
    bias_anchors: torch.Tensor,     # (K, H) from debias_vl Phase A
    anti_anchors: torch.Tensor,     # (K, H)
    cfg: CBDCConfig,
) -> tuple[FinBERTEncoder, torch.Tensor]:
    """
    CBDC text_iccv loop — iterative PGD + encoder training.

    Each epoch:
      1. PGD on target_text concept prompts
      2. S = z_adv_pos - z_adv_neg (confound direction per class-conditioned target)
      3. Class-specific match_loss + ck_loss → update layer 11 weights

    Returns:
        encoder:          fine-tuned FinBERTEncoder
        final_directions: (K, H)  last-epoch PGD-refined directions (for residual projection)
    """
    device = cfg.device
    n_cls = len(cls_text_groups)
    n_target = len(target_text)
    if n_target != n_cls:
        raise ValueError(
            f"RN50-style indexed match_loss expects len(target_text) == len(cls_text_groups); "
            f"got {n_target} vs {n_cls}"
        )

    print(f"\n=== Phase B+C: CBDC text_iccv training ===")
    print(f"  n_epochs={cfg.n_epochs} | lr={cfg.lr} | "
          f"PGD: ε={cfg.epsilon} steps={cfg.n_pgd_steps} restarts={cfg.num_samples}")

    # Tokenize concept prompts once
    cls_flat_text = flatten_prompt_groups(cls_text_groups)
    cls_enc = encoder.tokenize(cls_flat_text)
    target_enc = encoder.tokenize(target_text)
    cls_ids  = cls_enc["input_ids"].to(device)
    cls_mask = cls_enc["attention_mask"].to(device)
    target_ids = target_enc["input_ids"].to(device)
    target_mask = target_enc["attention_mask"].to(device)

    # keep_cb for L_s (finance semantics we want to preserve)
    with torch.no_grad():
        keep_cb = encoder.encode_text(keep_text).to(device)

    # Set up trainable layer 11
    layer_11 = encoder.backbone.encoder.layer[-1]
    for p in layer_11.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(layer_11.parameters(), lr=cfg.lr, weight_decay=1e-4)

    n_trainable = sum(p.numel() for p in layer_11.parameters())
    print(f"  Trainable params (layer 11): {n_trainable:,}")

    last_S = None

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
        for p in layer_11.parameters():
            p.requires_grad_(False)

        z_adv_pos, z_adv_neg = _pgd_bipolar(
            h_target,
            target_mask,
            z_orig,
            bias_anchors,
            anti_anchors,
            keep_cb,
            encoder, cfg,
        )

        # --- 3. Forward adversarial through layer 11 (frozen) to get S ---
        with torch.no_grad():
            S = z_adv_pos - z_adv_neg   # (num_samples * N_test, H)
        last_S = S.detach().cpu()

        # --- 4. Unfreeze layer 11, compute cls_em WITH gradient ---
        for p in layer_11.parameters():
            p.requires_grad_(True)

        cls_em = _encode_class_prototypes(encoder, cls_ids, cls_mask, cls_group_sizes)

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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                alignment = (S.to(device) @ cls_em.detach().T).pow(2).mean().item()
            print(f"  Epoch {epoch+1}/{cfg.n_epochs}: "
                  f"match={match_loss.item():.4f} ck={ck_loss.item():.4f} "
                  f"total={total_loss.item():.4f} alignment={alignment:.6f}")

    # Freeze after training
    for p in layer_11.parameters():
        p.requires_grad_(False)
    encoder.backbone.eval()

    # Extract final directions from last S via SVD
    S_centered = last_S - last_S.mean(0)
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
            cls_group_sizes,
            normalize=True,
        )

    return encoder, final_directions, final_cls_em


# ═══════════════════════════════════════════════════════════════════════════
# Re-encode all splits with fine-tuned encoder
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def reencode_splits(encoder: FinBERTEncoder, device: str, suffix: str = "cbdc"):
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
        n_bias_dirs=args.n_bias_dirs,
        lambda_reg=args.lambda_reg,
        device=device,
    )

    # ---- Load encoder -------------------------------------------------------
    model_name = os.environ.get("MODEL_NAME", "ProsusAI/finbert")
    encoder = FinBERTEncoder(model_name=model_name, device=device)

    # ---- Phase A: debias_vl map discovery -----------------------------------
    P_debias, bias_anchors, anti_anchors = discover_confound_map(encoder, cfg)

    # Save debias_vl projection matrix
    p_path = os.path.join(CACHE_DIR, "debias_vl_P.pt")
    torch.save(P_debias.cpu(), p_path)
    print(f"  debias_vl projection -> {p_path}")

    # ---- Phase B+C: CBDC text_iccv training ---------------------------------
    encoder, final_directions, final_cls_prototypes = text_iccv(encoder, bias_anchors, anti_anchors, cfg)

    # Save outputs
    dir_path = os.path.join(CACHE_DIR, "cbdc_directions.pt")
    torch.save(final_directions, dir_path)
    print(f"  CBDC directions ({tuple(final_directions.shape)}) -> {dir_path}")

    sent_path = os.path.join(CACHE_DIR, "sentiment_prototypes.pt")
    torch.save(final_cls_prototypes, sent_path)
    print(f"  Sentiment prototypes ({tuple(final_cls_prototypes.shape)}) -> {sent_path}")

    ckpt_path = os.path.join(CACHE_DIR, "encoder_cbdc.pt")
    torch.save(encoder.backbone.encoder.layer[-1].state_dict(), ckpt_path)
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
