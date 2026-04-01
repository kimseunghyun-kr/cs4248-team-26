"""
Phase 2: materialize the three derived methods with isolated artifacts.

Conditions:
  D1 (debias_vl)        : closed-form debias_vl projection on raw embeddings
  D2 (CBDC)             : pure prompt-driven CBDC training
  D3 (debias_vl->CBDC)  : debias_vl discovery feeding CBDC training

Each condition writes only into its own method-scoped directory under:
  cache/conditions/<condition-slug>/

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
from pipeline.clean import project_out
from pipeline.artifacts import (
    condition_artifact_path,
    condition_split_path,
    ensure_condition_dir,
    raw_split_path,
)
from cbdc.prompts import (
    encode_all_prompts,
    flatten_prompt_groups,
    get_cbdc_prompt_bank,
    get_combined_prompt_bank,
    get_debias_vl_prompt_bank,
    pool_prompt_group_embeddings,
)


_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)

D1_LABEL = "D1 (debias_vl)"
D2_LABEL = "D2 (CBDC)"
D3_LABEL = "D3 (debias_vl->CBDC)"


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _clone_split_payload(raw_data: dict, embeddings: torch.Tensor) -> dict:
    payload = dict(raw_data)
    payload["embeddings"] = embeddings.cpu()
    return payload


def _load_raw_split(split: str) -> dict:
    path = raw_split_path(CACHE_DIR, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required cached split not found: {path}. Run data/embed.py first."
        )
    return torch.load(path, map_location="cpu")


def _save_condition_split(condition_label: str, split: str, payload: dict) -> None:
    ensure_condition_dir(CACHE_DIR, condition_label)
    path = condition_split_path(CACHE_DIR, condition_label, split)
    torch.save(payload, path)
    print(f"  {condition_label} {split}: saved -> {path}")


def _load_encoder(model_name: str, device: str, tokenizer_name: str | None) -> TransformerEncoder:
    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer
        print(f"Loading custom tokenizer from '{tokenizer_name}' ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return TransformerEncoder(model_name=model_name, device=device, tokenizer=tokenizer)


# ═══════════════════════════════════════════════════════════════════════════
# debias_vl discovery
# ═══════════════════════════════════════════════════════════════════════════

def _get_A(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return z_i @ z_i.T + z_j @ z_j.T - z_i @ z_j.T - z_j @ z_i.T


def _get_M(embeddings: torch.Tensor, s_pairs: list[tuple[int, int]]) -> torch.Tensor:
    d = embeddings.shape[1]
    M = torch.zeros((d, d), device=embeddings.device)
    for s in s_pairs:
        M += _get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(s_pairs)


def _get_proj_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    _, _, V = torch.svd(embeddings)
    basis = V
    proj_sup = basis @ torch.inverse(basis.T @ basis) @ basis.T
    return torch.eye(proj_sup.shape[0], device=embeddings.device) - proj_sup


def _instantiate_anchor_poles(
    spurious_cb: torch.Tensor,
    topics: list[str],
    svd_dirs: torch.Tensor,
    phrases_per_side: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
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

    return torch.stack(bias_anchors, dim=0), torch.stack(anti_anchors, dim=0), anchor_info


def discover_confound_map(
    encoder: TransformerEncoder,
    cfg: CBDCConfig,
    prompt_bank: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Run the debias_vl discovery stage and return its method-specific artifacts."""
    print("\n=== debias_vl confound map discovery ===")

    prompts = encode_all_prompts(encoder, prompt_bank)
    spurious_cb = prompts["spurious_cb"]
    candidate_cb = prompts["candidate_cb"]
    s_pairs = prompt_bank["S_pairs"]
    topics = prompt_bank["topics"]

    print(f"  spurious_cb:  {tuple(spurious_cb.shape)}")
    print(f"  candidate_cb: {tuple(candidate_cb.shape)}")
    print(f"  S_pairs: {len(s_pairs)}")
    source = "mined topic phrases" if prompt_bank.get("using_mined_topics") else "fallback topic bank"
    print(f"  topic source: {source} ({len(topics)} topics)")
    if prompt_bank.get("topic_metadata"):
        preview = ", ".join(topic["topic"] for topic in prompt_bank["topic_metadata"][:8])
        print(f"  topic preview: {preview}")
    if prompt_bank.get("mining_error"):
        print(f"  mining fallback reason: {prompt_bank['mining_error']}")

    P0 = _get_proj_matrix(spurious_cb)
    M = _get_M(candidate_cb, s_pairs)
    H = candidate_cb.shape[1]
    G = cfg.lambda_reg * M + torch.eye(H, device=candidate_cb.device)
    P_debias = P0 @ torch.inverse(G)

    confound_matrix = torch.eye(H, device=P_debias.device) - P_debias
    _, singular_values, Vh = torch.linalg.svd(confound_matrix, full_matrices=False)
    svd_dirs = F.normalize(Vh[:cfg.n_bias_dirs], dim=-1)
    bias_anchors, anti_anchors, anchor_info = _instantiate_anchor_poles(
        spurious_cb,
        topics,
        svd_dirs,
        phrases_per_side=cfg.pole_phrases_per_side,
    )

    print(f"  Top-{cfg.n_bias_dirs} SVD singular values: {singular_values[:cfg.n_bias_dirs].tolist()}")
    print(f"  bias_anchors: {tuple(bias_anchors.shape)}")

    cls_cb = prompts.get("cls_cb")
    if cls_cb is not None:
        for i in range(cfg.n_bias_dirs):
            cos_cls = (bias_anchors[i] @ cls_cb.T).abs().max().item()
            print(f"    anchor {i}: max|cos(anchor, cls_em)|={cos_cls:.4f}")

    return P_debias, svd_dirs, bias_anchors, anti_anchors, anchor_info


# ═══════════════════════════════════════════════════════════════════════════
# Pure/prompt-driven CBDC
# ═══════════════════════════════════════════════════════════════════════════

def _pgd_bipolar(
    h_target: torch.Tensor,
    attention_mask: torch.Tensor,
    z_orig: torch.Tensor,
    bias_anchors: torch.Tensor,
    anti_anchors: torch.Tensor,
    keep_cb: torch.Tensor,
    encoder: TransformerEncoder,
    cfg: CBDCConfig,
    sentiment_protos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = cfg.device
    n_target = h_target.shape[0]
    H = encoder.hidden_size

    bias_anchors = F.normalize(bias_anchors.to(device), dim=-1)
    anti_anchors = F.normalize(anti_anchors.to(device), dim=-1)
    keep_cb = F.normalize(keep_cb.to(device), dim=-1)

    sentiment_basis = None
    if sentiment_protos is not None and cfg.sent_orthogonal_pgd:
        sentiment_basis = F.normalize(sentiment_protos.to(device), dim=-1)
        Q, _ = torch.linalg.qr(sentiment_basis.T)
        sentiment_basis = Q.T

    adv_pos_all = []
    adv_neg_all = []

    for restart in range(cfg.num_samples):
        if restart == 0:
            init_pos = torch.zeros(n_target, H, device=device)
            init_neg = torch.zeros(n_target, H, device=device)
        else:
            init_pos = (torch.rand(n_target, H, device=device) * 2 - 1) * cfg.random_eps
            init_neg = (torch.rand(n_target, H, device=device) * 2 - 1) * cfg.random_eps

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
                if sentiment_basis is not None:
                    grad = grad - (grad @ sentiment_basis.T) @ sentiment_basis
                delta.data += cfg.step_lr * grad.sign()
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)
        with torch.no_grad():
            adv_pos_all.append(
                encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta.detach())
            )

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
                if sentiment_basis is not None:
                    grad = grad - (grad @ sentiment_basis.T) @ sentiment_basis
                delta.data += cfg.step_lr * grad.sign()
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)
        with torch.no_grad():
            adv_neg_all.append(
                encoder.encode_with_delta_from_hidden(h_target, attention_mask, delta.detach())
            )

    return torch.cat(adv_pos_all, dim=0), torch.cat(adv_neg_all, dim=0)


def _encode_class_prototypes(
    encoder: TransformerEncoder,
    cls_ids: torch.Tensor,
    cls_mask: torch.Tensor,
    group_sizes: list[int],
) -> torch.Tensor:
    with torch.no_grad():
        h_cls = encoder.get_intermediate_features(cls_ids, cls_mask)
    cls_bank_em = encoder.encode_with_delta_from_hidden(
        h_cls,
        cls_mask,
        torch.zeros(len(cls_ids), encoder.hidden_size, device=cls_ids.device),
    )
    return pool_prompt_group_embeddings(cls_bank_em, group_sizes, normalize=True)


def _select_balanced_indices(
    labels: torch.Tensor,
    max_per_class: int,
    seed: int = 42,
) -> torch.Tensor:
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
    train_data = _load_raw_split("train")
    val_data = _load_raw_split("val")

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


def text_iccv(
    encoder: TransformerEncoder,
    bias_anchors: torch.Tensor,
    anti_anchors: torch.Tensor,
    cfg: CBDCConfig,
    prompt_bank: dict,
) -> tuple[TransformerEncoder, torch.Tensor, torch.Tensor]:
    """Run prompt-driven CBDC training for either D2 or D3."""
    device = cfg.device
    cls_text_groups = prompt_bank["cls_text_groups"]
    cls_group_sizes = prompt_bank["cls_group_sizes"]
    target_text = prompt_bank["target_text"]
    keep_text = prompt_bank["keep_text"]

    n_cls = len(cls_text_groups)
    n_target = len(target_text)
    if n_target != n_cls:
        raise ValueError(
            "CBDC match_loss expects len(target_text) == len(cls_text_groups)."
        )

    print("\n=== CBDC text_iccv training ===")
    print(
        f"  n_epochs={cfg.n_epochs} | lr={cfg.lr} | "
        f"PGD: epsilon={cfg.epsilon} steps={cfg.n_pgd_steps} restarts={cfg.num_samples}"
    )

    cls_flat_text = flatten_prompt_groups(cls_text_groups)
    cls_enc = encoder.tokenize(cls_flat_text)
    target_enc = encoder.tokenize(target_text)
    cls_ids = cls_enc["input_ids"].to(device)
    cls_mask = cls_enc["attention_mask"].to(device)
    target_ids = target_enc["input_ids"].to(device)
    target_mask = target_enc["attention_mask"].to(device)

    with torch.no_grad():
        keep_cb = encoder.encode_text(keep_text).to(device)

    sentiment_protos = None
    if cfg.sent_orthogonal_pgd:
        with torch.no_grad():
            init_cls_em = _encode_class_prototypes(encoder, cls_ids, cls_mask, cls_group_sizes)
        sentiment_protos = init_cls_em.detach().to(device)
        print("  Sentiment-orthogonal PGD: ON")

    layer_tail = encoder._get_transformer_layers()[-1]
    for p in layer_tail.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(layer_tail.parameters(), lr=cfg.lr, weight_decay=1e-4)

    selector_data = _prepare_selector_data(cfg)
    best_selector_f1 = float("-inf")
    best_epoch = None
    best_state = None

    for epoch in tqdm(range(cfg.n_epochs), desc="text_iccv"):
        with torch.no_grad():
            h_target = encoder.get_intermediate_features(target_ids, target_mask)
            z_orig = encoder.encode_with_delta_from_hidden(
                h_target,
                target_mask,
                torch.zeros(n_target, encoder.hidden_size, device=device),
            )

        for p in layer_tail.parameters():
            p.requires_grad_(False)

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

        with torch.no_grad():
            S = z_adv_pos - z_adv_neg

        for p in layer_tail.parameters():
            p.requires_grad_(True)

        cls_em = _encode_class_prototypes(encoder, cls_ids, cls_mask, cls_group_sizes)
        if sentiment_protos is not None:
            sentiment_protos = cls_em.detach().clone()

        match_loss = torch.tensor(0.0, device=device)
        for c in range(n_target):
            match_loss += (S[c::n_target].to(device) @ cls_em[c:c+1].T).pow(2).mean()
        match_loss = match_loss * cfg.up_scale / n_target

        ck_loss = l_ck(
            bias_anchors.to(device),
            anti_anchors.to(device),
            cls_em,
            scale=cfg.up_scale,
        )

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
            selector_msg = ""
            if selector_f1 is not None:
                selector_msg = f" selector_f1={selector_f1:.4f}"
            print(
                f"  Epoch {epoch+1}/{cfg.n_epochs}: "
                f"match={match_loss.item():.4f} ck={ck_loss.item():.4f} "
                f"total={total_loss.item():.4f}{selector_msg}"
            )

    if best_state is not None:
        layer_tail.load_state_dict(best_state)
        print(
            f"\n  Restored best CBDC epoch: {best_epoch}/{cfg.n_epochs} "
            f"(selector_f1={best_selector_f1:.4f})"
        )

    for p in layer_tail.parameters():
        p.requires_grad_(False)
    encoder.backbone.eval()

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

    S_centered = best_S - best_S.mean(0)
    _, _, Vh = torch.linalg.svd(S_centered, full_matrices=False)
    final_directions = Vh[:cfg.n_bias_dirs]

    with torch.no_grad():
        cls_bank_final = encoder.encode_text(cls_flat_text).cpu()
        final_cls_em = pool_prompt_group_embeddings(
            cls_bank_final,
            cls_group_sizes,
            normalize=True,
        )

    return encoder, final_directions.cpu(), final_cls_em.cpu()


@torch.no_grad()
def reencode_splits_to_condition(
    encoder: TransformerEncoder,
    device: str,
    condition_label: str,
) -> None:
    ensure_condition_dir(CACHE_DIR, condition_label)
    for split in ["train", "val", "test"]:
        raw_data = _load_raw_split(split)
        ids = raw_data["input_ids"]
        mask = raw_data["attention_mask"]

        batch_size = 128
        all_z = []
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size].to(device)
            batch_mask = mask[i:i+batch_size].to(device)
            z = encoder.encode_ids(batch_ids, batch_mask)
            all_z.append(z.cpu())

        z_new = torch.cat(all_z, dim=0)
        _save_condition_split(condition_label, split, _clone_split_payload(raw_data, z_new))


def _save_prompt_bank(condition_label: str, prompt_bank: dict) -> None:
    ensure_condition_dir(CACHE_DIR, condition_label)
    _save_json(
        condition_artifact_path(CACHE_DIR, condition_label, "prompt_bank.json"),
        prompt_bank,
    )


def materialize_debias_vl_condition(
    encoder: TransformerEncoder,
    cfg: CBDCConfig,
    prompt_bank: dict,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"Materializing {D1_LABEL}")
    print(f"{'=' * 72}")

    ensure_condition_dir(CACHE_DIR, D1_LABEL)
    _save_prompt_bank(D1_LABEL, prompt_bank)

    P_debias, svd_dirs, bias_anchors, anti_anchors, anchor_info = discover_confound_map(
        encoder,
        cfg,
        prompt_bank,
    )

    torch.save(P_debias.cpu(), condition_artifact_path(CACHE_DIR, D1_LABEL, "debias_vl_P.pt"))
    torch.save(svd_dirs.cpu(), condition_artifact_path(CACHE_DIR, D1_LABEL, "debias_vl_directions.pt"))
    torch.save(bias_anchors.cpu(), condition_artifact_path(CACHE_DIR, D1_LABEL, "bias_anchors.pt"))
    torch.save(anti_anchors.cpu(), condition_artifact_path(CACHE_DIR, D1_LABEL, "anti_anchors.pt"))
    _save_json(
        condition_artifact_path(CACHE_DIR, D1_LABEL, "anchor_poles.json"),
        {"anchor_info": anchor_info},
    )

    for split in ["train", "val", "test"]:
        raw_data = _load_raw_split(split)
        z_clean = project_out(raw_data["embeddings"], P_debias.cpu())
        _save_condition_split(D1_LABEL, split, _clone_split_payload(raw_data, z_clean))


def materialize_cbdc_condition(
    condition_label: str,
    encoder: TransformerEncoder,
    cfg: CBDCConfig,
    cbdc_prompt_bank: dict,
    bias_anchors: torch.Tensor,
    anti_anchors: torch.Tensor,
    prompt_bank_payload: dict,
    extra_artifacts: dict[str, torch.Tensor] | None = None,
    extra_json: dict | None = None,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"Materializing {condition_label}")
    print(f"{'=' * 72}")

    ensure_condition_dir(CACHE_DIR, condition_label)
    _save_prompt_bank(condition_label, prompt_bank_payload)

    torch.save(bias_anchors.cpu(), condition_artifact_path(CACHE_DIR, condition_label, "bias_anchors.pt"))
    torch.save(anti_anchors.cpu(), condition_artifact_path(CACHE_DIR, condition_label, "anti_anchors.pt"))

    if extra_artifacts:
        for filename, tensor in extra_artifacts.items():
            torch.save(tensor.cpu(), condition_artifact_path(CACHE_DIR, condition_label, filename))
    if extra_json:
        for filename, payload in extra_json.items():
            _save_json(condition_artifact_path(CACHE_DIR, condition_label, filename), payload)

    encoder, final_directions, final_cls_prototypes = text_iccv(
        encoder,
        bias_anchors,
        anti_anchors,
        cfg,
        cbdc_prompt_bank,
    )

    torch.save(
        final_directions.cpu(),
        condition_artifact_path(CACHE_DIR, condition_label, "cbdc_directions.pt"),
    )
    torch.save(
        final_cls_prototypes.cpu(),
        condition_artifact_path(CACHE_DIR, condition_label, "class_prompt_prototypes.pt"),
    )
    torch.save(
        encoder._get_transformer_layers()[-1].state_dict(),
        condition_artifact_path(CACHE_DIR, condition_label, "encoder_cbdc.pt"),
    )

    reencode_splits_to_condition(encoder, cfg.device, condition_label)


def main():
    parser = argparse.ArgumentParser(
        description="Materialize D1 / D2 / D3 with isolated method-scoped artifacts."
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--n_pgd_steps", type=int, default=20)
    parser.add_argument("--step_lr", type=float, default=0.0037)
    parser.add_argument("--keep_weight", type=float, default=0.92)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--random_eps", type=float, default=0.22)
    parser.add_argument("--n_bias_dirs", type=int, default=4)
    parser.add_argument("--lambda_reg", type=float, default=1000.0)
    parser.add_argument("--up_scale", type=float, default=100.0)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--selector_train_per_class", type=int, default=512)
    parser.add_argument("--selector_batch_size", type=int, default=128)
    parser.add_argument("--use_static_topics", action="store_true",
                        help="Disable topic mining for debias_vl and use the static fallback bank.")
    parser.add_argument("--mine_max_topics", type=int, default=32)
    parser.add_argument("--mine_min_doc_freq", type=int, default=20)
    parser.add_argument("--mine_max_doc_freq_ratio", type=float, default=0.20)
    parser.add_argument("--pole_phrases_per_side", type=int, default=4)
    parser.add_argument("--refresh_mined_topics", action="store_true",
                        help="Ignore cached mined_topics.json and mine the topic bank again.")
    parser.add_argument("--no_sent_orthogonal_pgd", action="store_true",
                        help="Disable sentiment-orthogonal PGD gradient projection.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(CACHE_DIR, exist_ok=True)

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
        sent_orthogonal_pgd=not (args.no_sent_orthogonal_pgd or os.environ.get("NO_SENT_ORTHOGONAL_PGD") == "1"),
        device=device,
    )

    model_name = os.environ.get("MODEL_NAME", "bert-base-uncased")
    tokenizer_name = os.environ.get("TOKENIZER_NAME")
    text_unit = os.environ.get("TEXT_UNIT", cfg.text_unit)

    debias_prompt_bank = get_debias_vl_prompt_bank(
        tokenizer=None,
        cache_dir=CACHE_DIR,
        use_mined_topics=cfg.use_mined_topics,
        max_topics=cfg.mine_max_topics,
        min_doc_freq=cfg.mine_min_doc_freq,
        max_doc_freq_ratio=cfg.mine_max_doc_freq_ratio,
        force_refresh=args.refresh_mined_topics,
        text_unit=text_unit,
    )
    cbdc_prompt_bank = get_cbdc_prompt_bank(text_unit=text_unit)
    combined_prompt_bank = get_combined_prompt_bank(
        tokenizer=None,
        cache_dir=CACHE_DIR,
        use_mined_topics=cfg.use_mined_topics,
        max_topics=cfg.mine_max_topics,
        min_doc_freq=cfg.mine_min_doc_freq,
        max_doc_freq_ratio=cfg.mine_max_doc_freq_ratio,
        force_refresh=args.refresh_mined_topics,
        text_unit=text_unit,
    )

    encoder_d1 = _load_encoder(model_name=model_name, device=device, tokenizer_name=tokenizer_name)
    materialize_debias_vl_condition(
        encoder=encoder_d1,
        cfg=cfg,
        prompt_bank=debias_prompt_bank,
    )

    encoder_d2 = _load_encoder(model_name=model_name, device=device, tokenizer_name=tokenizer_name)
    prompt_encodings_d2 = encode_all_prompts(encoder_d2, cbdc_prompt_bank)
    materialize_cbdc_condition(
        condition_label=D2_LABEL,
        encoder=encoder_d2,
        cfg=cfg,
        cbdc_prompt_bank=cbdc_prompt_bank,
        bias_anchors=prompt_encodings_d2["bias_pole_a_cb"],
        anti_anchors=prompt_encodings_d2["bias_pole_b_cb"],
        prompt_bank_payload=cbdc_prompt_bank,
    )

    encoder_d3 = _load_encoder(model_name=model_name, device=device, tokenizer_name=tokenizer_name)
    P_debias_d3, svd_dirs_d3, bias_anchors_d3, anti_anchors_d3, anchor_info_d3 = discover_confound_map(
        encoder_d3,
        cfg,
        debias_prompt_bank,
    )
    materialize_cbdc_condition(
        condition_label=D3_LABEL,
        encoder=encoder_d3,
        cfg=cfg,
        cbdc_prompt_bank=cbdc_prompt_bank,
        bias_anchors=bias_anchors_d3,
        anti_anchors=anti_anchors_d3,
        prompt_bank_payload=combined_prompt_bank,
        extra_artifacts={
            "debias_vl_P.pt": P_debias_d3,
            "debias_vl_directions.pt": svd_dirs_d3,
        },
        extra_json={
            "anchor_poles.json": {"anchor_info": anchor_info_d3},
        },
    )

    print("\nPhase 2 materialization complete.")


if __name__ == "__main__":
    main()
