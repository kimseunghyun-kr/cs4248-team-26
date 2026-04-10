from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor, CLIPModel

from config import get_model_name
from losses import l_bias_contrastive, l_ck, l_semantic_preservation
from .prompts import PersonalizationPromptBank


TEMPERATURE = 100.0


@dataclass
class TextICCVArtifacts:
    adapter_state: dict[str, torch.Tensor]
    directions: torch.Tensor
    class_embeddings: torch.Tensor
    bias_embeddings: torch.Tensor
    anti_embeddings: torch.Tensor
    losses: list[dict[str, float]]
    source_model_name: str

    def to_json(self) -> dict[str, Any]:
        return {
            "source_model_name": self.source_model_name,
            "losses": self.losses,
            "direction_shape": list(self.directions.shape),
            "class_embedding_shape": list(self.class_embeddings.shape),
            "bias_embedding_shape": list(self.bias_embeddings.shape),
            "anti_embedding_shape": list(self.anti_embeddings.shape),
        }


@dataclass
class ImageICCVArtifacts:
    adapter_state: dict[str, torch.Tensor]
    losses: list[dict[str, float]]
    source_model_name: str
    text_source_model_name: str
    class_embeddings: torch.Tensor

    def to_json(self) -> dict[str, Any]:
        return {
            "source_model_name": self.source_model_name,
            "text_source_model_name": self.text_source_model_name,
            "losses": self.losses,
            "class_embedding_shape": list(self.class_embeddings.shape),
        }


class EmbeddingTargetAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(features.float()), dim=-1)


class MultimodalEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = get_model_name(model_name)
        self.device = device
        self.load_dtype = self._resolve_load_dtype(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model, self.backend = self._load_model()
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.hidden_size = self._resolve_hidden_size()

    @staticmethod
    def _resolve_load_dtype(model_name: str) -> torch.dtype:
        lower = model_name.lower()
        if "gemma-4" in lower or "gemma4" in lower:
            return torch.bfloat16
        return torch.float32

    def _load_model(self) -> tuple[nn.Module, str]:
        load_kwargs = {
            "dtype": self.load_dtype,
            "low_cpu_mem_usage": True,
        }
        try:
            return (
                AutoModelForImageTextToText.from_pretrained(self.model_name, **load_kwargs),
                "vlm",
            )
        except Exception as vlm_exc:
            try:
                return (
                    CLIPModel.from_pretrained(self.model_name, **load_kwargs),
                    "clip",
                )
            except Exception as clip_exc:
                raise RuntimeError(
                    f"Could not load '{self.model_name}' as a VLM or CLIP-style backbone."
                ) from clip_exc if clip_exc is not None else vlm_exc

    def _resolve_hidden_size(self) -> int:
        if self.backend == "clip":
            return int(self.model.config.projection_dim)
        cfg = getattr(self.model, "config", None)
        if cfg is not None and hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return int(cfg.text_config.hidden_size)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise AttributeError(f"Could not determine hidden size for {self.model_name}.")

    @staticmethod
    def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, positions]

    @staticmethod
    def _unwrap_feature_output(features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        raise TypeError(f"Unsupported feature output type: {type(features).__name__}")

    def _prepare_vlm_image_inputs(self, images: list[Any]) -> dict[str, Any]:
        if self.backend == "clip":
            return self.processor(images=images, return_tensors="pt")

        processor_name = type(self.processor).__name__.lower()
        if "qwen" in processor_name:
            image_token = getattr(self.processor, "image_token", "<|image_pad|>")
            return self.processor(
                images=images,
                text=[image_token] * len(images),
                return_tensors="pt",
            )
        return self.processor(images=images, return_tensors="pt")

    @torch.no_grad()
    def encode_text(self, texts: list[str], max_length: int = 128) -> torch.Tensor:
        if self.backend == "clip":
            inputs = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = self._move_batch(inputs, self.device)
            text_features = self._unwrap_feature_output(self.model.get_text_features(**inputs))
            return F.normalize(text_features.float(), dim=-1)

        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = self._move_batch(inputs, self.device)
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        pooled = self._pool_last_token(outputs.hidden_states[-1], inputs["attention_mask"])
        return F.normalize(pooled.float(), dim=-1)

    @torch.no_grad()
    def encode_images(self, images: list[Any]) -> torch.Tensor:
        inputs = self._prepare_vlm_image_inputs(images)
        inputs = self._move_batch(inputs, self.device)

        if self.backend == "clip":
            image_features = self._unwrap_feature_output(self.model.get_image_features(**inputs))
            return F.normalize(image_features.float(), dim=-1)

        call_kwargs: dict[str, Any] = {"pixel_values": inputs["pixel_values"]}
        if "image_grid_thw" in inputs:
            call_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
        if "image_position_ids" in inputs:
            call_kwargs["image_position_ids"] = inputs["image_position_ids"]

        outputs = self.model.get_image_features(**call_kwargs)
        pooled = outputs.pooler_output
        if isinstance(pooled, (list, tuple)):
            vectors = [tensor.float().mean(dim=0) if tensor.dim() > 1 else tensor.float() for tensor in pooled]
            pooled = torch.stack(vectors, dim=0)
        elif pooled.dim() == 3:
            pooled = pooled.float().mean(dim=1)
        else:
            pooled = pooled.float()
        return F.normalize(pooled, dim=-1)


def _aggregate_axis_embeddings(
    encode_text_fn,
    prompt_bank: PersonalizationPromptBank,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_bank = []
    neg_bank = []
    for axis in prompt_bank.nuisance_axes:
        pos_bank.append(encode_text_fn(axis.positive_prompts, max_length=max_length).mean(dim=0))
        neg_bank.append(encode_text_fn(axis.negative_prompts, max_length=max_length).mean(dim=0))
    return (
        F.normalize(torch.stack(pos_bank, dim=0), dim=-1),
        F.normalize(torch.stack(neg_bank, dim=0), dim=-1),
    )


def _clamp_linf(z_adv: torch.Tensor, z_nat: torch.Tensor, bound: float) -> torch.Tensor:
    return torch.max(torch.min(z_adv, z_nat + bound), z_nat - bound)


def perturb_text_feature_bipolar(
    z: torch.Tensor,
    target_model: EmbeddingTargetAdapter,
    bias_a: torch.Tensor,
    bias_b: torch.Tensor,
    keep_axes: torch.Tensor | None = None,
    att_bnd: float = 1.0,
    att_stp: float = 0.0037,
    att_itr: int = 20,
    random: float = 0.22,
    num_samples: int = 8,
    keep_weight: float = 0.92,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = z.device
    z_nat = z.detach().clone().to(device)

    with torch.no_grad():
        ori = target_model(z_nat)

    z_adv_list = []
    z_adv_list2 = []
    bias_a = F.normalize(bias_a.to(device), dim=-1)
    bias_b = F.normalize(bias_b.to(device), dim=-1)
    keep_axes = None if keep_axes is None else F.normalize(keep_axes.to(device), dim=-1)

    for num in range(num_samples):
        if num == 0:
            base = z_nat
        else:
            rand = torch.empty_like(z_nat).uniform_(-random, random)
            base = _clamp_linf(z_nat + rand, z_nat, bound=att_bnd)

        for push_toward_a, bucket in ((True, z_adv_list), (False, z_adv_list2)):
            z_adv = base.clone().detach().requires_grad_(True)
            for _ in range(att_itr):
                target_model.zero_grad()
                adv_feat = target_model(z_adv)
                att_loss = l_bias_contrastive(
                    adv_feat,
                    bias_a,
                    bias_b,
                    push_toward_a=push_toward_a,
                    temperature=TEMPERATURE,
                )
                if keep_axes is not None:
                    keep_loss = l_semantic_preservation(adv_feat, ori, keep_axes)
                    loss = att_loss * (1.0 - keep_weight) - keep_loss * keep_weight
                else:
                    loss = att_loss
                loss.backward()
                with torch.no_grad():
                    step = z_adv.grad.detach().sign() * att_stp
                    z_adv = _clamp_linf(z_adv + step, base, bound=att_bnd).detach().requires_grad_(True)
            bucket.append(z_adv.detach())

    return torch.cat(z_adv_list, dim=0), torch.cat(z_adv_list2, dim=0)


def bafa_img_iccv(
    z: torch.Tensor,
    target_model: EmbeddingTargetAdapter,
    bias_txt: torch.Tensor,
    debias_txt: torch.Tensor,
    bound: float = 0.1,
    step: float = 1e-2,
    iters: int = 20,
    l_scale: float = 100.0,
    lambda_: float = 1.0,
    rand_eps: float = 0.0,
) -> torch.Tensor:
    device = z.device
    z_nat = z.detach().clone().to(device)

    if rand_eps > 0:
        rand_perturb = torch.empty_like(z_nat).uniform_(-rand_eps, rand_eps)
        z_init = _clamp_linf(z_nat + rand_perturb, z_nat, bound=bound)
    else:
        z_init = z_nat

    z_adv = z_init.clone().detach().requires_grad_(True)

    with torch.no_grad():
        out_feat = target_model(z_nat)
        cls_d = l_scale * (out_feat @ debias_txt.T)
        debias_cls_predict = torch.argmax(cls_d, dim=-1)

    for _ in range(iters):
        target_model.zero_grad()
        adv_feat = target_model(z_adv)
        adv_loss = F.cross_entropy(l_scale * (adv_feat @ bias_txt.T), debias_cls_predict)
        adv_loss_keep = F.cross_entropy(l_scale * (adv_feat @ debias_txt.T), debias_cls_predict)
        loss = adv_loss - lambda_ * adv_loss_keep
        loss.backward()
        with torch.no_grad():
            step_dir = z_adv.grad.detach().sign()
            z_adv = _clamp_linf(z_adv + step_dir * step, z_nat, bound=bound).detach().requires_grad_(True)

    return z_adv.detach()


def run_text_iccv(
    text_source,
    prompt_bank: PersonalizationPromptBank,
    epochs: int,
    lr: float,
    epsilon: float,
    step_lr: float,
    n_steps: int,
    num_restarts: int,
    random_eps: float,
    keep_weight: float,
    n_bias_dirs: int,
    max_length: int,
) -> TextICCVArtifacts:
    device = text_source.device
    target_model = EmbeddingTargetAdapter(text_source.hidden_size).to(device)
    optimizer = torch.optim.AdamW(target_model.parameters(), lr=lr)

    z_cls = text_source.encode_text(prompt_bank.class_prompts, max_length=max_length).to(device)
    z_keep = text_source.encode_text(prompt_bank.keep_prompts, max_length=max_length).to(device)
    z_target = text_source.encode_text(prompt_bank.target_prompts, max_length=max_length).to(device)
    z_bias, z_anti = _aggregate_axis_embeddings(text_source.encode_text, prompt_bank, max_length=max_length)
    z_bias = z_bias.to(device)
    z_anti = z_anti.to(device)

    history: list[dict[str, float]] = []
    final_s = None
    final_cls = None
    final_bias = None
    final_anti = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        cls_em = target_model(z_cls)
        bias_cb = target_model(z_bias)
        anti_cb = target_model(z_anti)
        with torch.no_grad():
            keep_em = target_model(z_keep)
            bias_cb_pgd = target_model(z_bias)
            anti_cb_pgd = target_model(z_anti)

        z_adv_set1, z_adv_set2 = perturb_text_feature_bipolar(
            z_target,
            target_model,
            bias_cb_pgd,
            anti_cb_pgd,
            keep_axes=keep_em,
            att_bnd=epsilon,
            att_stp=step_lr,
            att_itr=n_steps,
            random=random_eps,
            num_samples=num_restarts,
            keep_weight=keep_weight,
        )

        adv_cb_set1 = target_model(z_adv_set1)
        adv_cb_set2 = target_model(z_adv_set2)
        s_delta = adv_cb_set1 - adv_cb_set2

        match_loss = (s_delta @ cls_em.T).pow(2).mean()
        ck_loss = l_ck(bias_cb, anti_cb, cls_em)
        loss = match_loss + ck_loss
        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "match_loss": float(match_loss.item()),
                "ck_loss": float(ck_loss.item()),
            }
        )

        final_s = s_delta.detach()
        final_cls = cls_em.detach()
        final_bias = bias_cb.detach()
        final_anti = anti_cb.detach()

    centered = final_s - final_s.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    k = max(1, min(n_bias_dirs, vh.shape[0]))

    return TextICCVArtifacts(
        adapter_state={key: value.detach().cpu() for key, value in target_model.state_dict().items()},
        directions=F.normalize(vh[:k], dim=-1).cpu(),
        class_embeddings=final_cls.cpu(),
        bias_embeddings=final_bias.cpu(),
        anti_embeddings=final_anti.cpu(),
        losses=history,
        source_model_name=text_source.model_name,
    )


def _load_text_adapter_if_compatible(
    hidden_size: int,
    text_artifacts: TextICCVArtifacts | None,
    model_name: str,
    device: str,
) -> EmbeddingTargetAdapter:
    adapter = EmbeddingTargetAdapter(hidden_size).to(device)
    if text_artifacts is None:
        return adapter
    if text_artifacts.source_model_name != model_name:
        return adapter
    adapter.load_state_dict(text_artifacts.adapter_state)
    return adapter


def run_img_iccv(
    vlm: MultimodalEmbedder,
    prompt_bank: PersonalizationPromptBank,
    instance_loader,
    epochs: int,
    lr: float,
    bound: float,
    step: float,
    iters: int,
    l_scale: float,
    lambda_: float,
    rand_eps: float,
    max_length: int,
    text_artifacts: TextICCVArtifacts | None = None,
) -> ImageICCVArtifacts:
    device = vlm.device
    hidden_size = vlm.hidden_size
    target_model = EmbeddingTargetAdapter(hidden_size).to(device)
    optimizer = torch.optim.AdamW(target_model.parameters(), lr=lr)

    text_target = _load_text_adapter_if_compatible(
        hidden_size=hidden_size,
        text_artifacts=text_artifacts,
        model_name=vlm.model_name,
        device=device,
    )
    text_target.eval()

    frozen_cls = vlm.encode_text(prompt_bank.class_prompts, max_length=max_length).to(device)
    bias_txt_cb = F.normalize(frozen_cls, dim=-1)
    with torch.no_grad():
        debias_txt_cb = text_target(frozen_cls)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0

        for batch in instance_loader:
            optimizer.zero_grad()
            z = vlm.encode_images(batch["images"]).to(device)
            img_cb = target_model(z)
            z_adv_bias = bafa_img_iccv(
                z,
                target_model,
                bias_txt=bias_txt_cb,
                debias_txt=debias_txt_cb,
                bound=bound,
                step=step,
                iters=iters,
                l_scale=l_scale,
                lambda_=lambda_,
                rand_eps=rand_eps,
            )
            adv_embed = target_model(z_adv_bias)
            s_delta = l_scale * (img_cb - adv_embed)
            loss = (s_delta @ debias_txt_cb.T).pow(2).mean()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        history.append(
            {
                "epoch": float(epoch),
                "loss": running_loss / max(1, batches),
            }
        )

    return ImageICCVArtifacts(
        adapter_state={key: value.detach().cpu() for key, value in target_model.state_dict().items()},
        losses=history,
        source_model_name=vlm.model_name,
        text_source_model_name=text_artifacts.source_model_name if text_artifacts else vlm.model_name,
        class_embeddings=debias_txt_cb.detach().cpu(),
    )


def write_iccv_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
