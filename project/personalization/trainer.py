from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os

import torch

from config import get_model_name, model_slug
from encoder import TransformerEncoder
from .data import ConceptMediaRecord, build_dataloaders, write_manifest
from .iccv import (
    ImageICCVArtifacts,
    MultimodalEmbedder,
    TextICCVArtifacts,
    run_img_iccv,
    run_text_iccv,
    write_iccv_json,
)
from .pgd import DiscoveryArtifacts, discover_nuisance_directions
from .prompts import PersonalizationPromptBank, build_prompt_bank


@dataclass
class PersonalizationConfig:
    model: str = "qwen25-3b"
    image_model: str | None = None
    tokenizer: str | None = None
    max_length: int = 128
    batch_size: int = 4
    num_workers: int = 0
    media_mode: str = "image"
    epsilon: float = 1.0
    n_pgd_steps: int = 20
    step_lr: float = 0.0037
    keep_weight: float = 0.92
    num_restarts: int = 8
    random_eps: float = 0.22
    n_bias_dirs: int = 4
    text_iters: int = 20
    img_iters: int = 5
    text_lr: float = 1e-3
    img_lr: float = 1e-3
    img_bound: float = 0.1
    img_step: float = 1e-2
    img_attack_steps: int = 10
    img_loss_scale: float = 100.0
    img_lambda: float = 1.0
    device: str = "cpu"


class PersonalizationTrainer:
    def __init__(self, cfg: PersonalizationConfig):
        self.cfg = cfg
        self.model_name = get_model_name(cfg.model)
        self.cache_slug = model_slug(cfg.model)
        self.image_model_name = get_model_name(cfg.image_model or cfg.model)
        self._encoder: TransformerEncoder | None = None
        self._vlm_cache: dict[str, MultimodalEmbedder] = {}
        self._tokenizer = None
        if cfg.tokenizer:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    @property
    def encoder(self) -> TransformerEncoder:
        if self._encoder is None:
            self._encoder = TransformerEncoder(
                model_name=self.model_name,
                device=self.cfg.device,
                tokenizer=self._tokenizer,
            )
        return self._encoder

    def get_vlm(self, model_name: str | None = None) -> MultimodalEmbedder:
        resolved_name = get_model_name(model_name or self.image_model_name)
        if resolved_name not in self._vlm_cache:
            self._vlm_cache[resolved_name] = MultimodalEmbedder(
                model_name=resolved_name,
                device=self.cfg.device,
            )
        return self._vlm_cache[resolved_name]

    @staticmethod
    def _prefers_multimodal_text_source(model_name: str) -> bool:
        lower = model_name.lower()
        return any(token in lower for token in ("qwen2.5-vl", "qwen2-vl", "gemma-4", "gemma4", "clip"))

    def get_text_source(self):
        if self._prefers_multimodal_text_source(self.model_name):
            return self.get_vlm(self.model_name)
        return self.encoder

    def build_prompt_bank(self, concept_token: str, class_name: str) -> PersonalizationPromptBank:
        return build_prompt_bank(concept_token=concept_token, class_name=class_name)

    def build_data(
        self,
        instance_dir: str,
        prompt_bank: PersonalizationPromptBank,
        class_dir: str | None = None,
    ) -> dict:
        return build_dataloaders(
            instance_dir=instance_dir,
            instance_prompt=prompt_bank.instance_prompt,
            class_dir=class_dir,
            class_prompt=prompt_bank.class_prompt if class_dir else None,
            media_mode=self.cfg.media_mode,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )

    def discover(self, prompt_bank: PersonalizationPromptBank) -> DiscoveryArtifacts:
        return discover_nuisance_directions(
            encoder=self.encoder,
            prompt_bank=prompt_bank,
            epsilon=self.cfg.epsilon,
            step_lr=self.cfg.step_lr,
            n_steps=self.cfg.n_pgd_steps,
            num_restarts=self.cfg.num_restarts,
            random_eps=self.cfg.random_eps,
            keep_weight=self.cfg.keep_weight,
            n_bias_dirs=self.cfg.n_bias_dirs,
        )

    def text_iccv(self, prompt_bank: PersonalizationPromptBank) -> TextICCVArtifacts:
        return run_text_iccv(
            text_source=self.get_text_source(),
            prompt_bank=prompt_bank,
            epochs=self.cfg.text_iters,
            lr=self.cfg.text_lr,
            epsilon=self.cfg.epsilon,
            step_lr=self.cfg.step_lr,
            n_steps=self.cfg.n_pgd_steps,
            num_restarts=self.cfg.num_restarts,
            random_eps=self.cfg.random_eps,
            keep_weight=self.cfg.keep_weight,
            n_bias_dirs=self.cfg.n_bias_dirs,
            max_length=self.cfg.max_length,
        )

    def img_iccv(
        self,
        prompt_bank: PersonalizationPromptBank,
        data_payload: dict,
        text_artifacts: TextICCVArtifacts | None = None,
    ) -> ImageICCVArtifacts:
        vlm = self.get_vlm(self.image_model_name)
        return run_img_iccv(
            vlm=vlm,
            prompt_bank=prompt_bank,
            instance_loader=data_payload["instance_loader"],
            epochs=self.cfg.img_iters,
            lr=self.cfg.img_lr,
            bound=self.cfg.img_bound,
            step=self.cfg.img_step,
            iters=self.cfg.img_attack_steps,
            l_scale=self.cfg.img_loss_scale,
            lambda_=self.cfg.img_lambda,
            rand_eps=self.cfg.random_eps,
            max_length=self.cfg.max_length,
            text_artifacts=text_artifacts,
        )

    def save_run(
        self,
        output_dir: str,
        prompt_bank: PersonalizationPromptBank,
        data_payload: dict,
        discovery: DiscoveryArtifacts | None = None,
        text_iccv_artifacts: TextICCVArtifacts | None = None,
        img_iccv_artifacts: ImageICCVArtifacts | None = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(asdict(self.cfg), handle, indent=2)

        with open(os.path.join(output_dir, "prompt_bank.json"), "w", encoding="utf-8") as handle:
            json.dump(prompt_bank.to_json(), handle, indent=2)

        instance_records: list[ConceptMediaRecord] = data_payload["instance_records"]
        write_manifest(os.path.join(output_dir, "instance_manifest.json"), instance_records)

        class_records: list[ConceptMediaRecord] = data_payload.get("class_records", [])
        if class_records:
            write_manifest(os.path.join(output_dir, "class_manifest.json"), class_records)

        if discovery is not None:
            torch.save(discovery.directions, os.path.join(output_dir, "pgd_directions.pt"))
            torch.save(discovery.bias_anchors, os.path.join(output_dir, "bias_anchors.pt"))
            torch.save(discovery.anti_anchors, os.path.join(output_dir, "anti_anchors.pt"))
            torch.save(discovery.target_embeddings, os.path.join(output_dir, "target_embeddings.pt"))

            discovery_meta = {
                "target_prompts": discovery.target_prompts,
                "axis_names": discovery.axis_names,
                "singular_values": discovery.singular_values,
            }
            with open(os.path.join(output_dir, "discovery.json"), "w", encoding="utf-8") as handle:
                json.dump(discovery_meta, handle, indent=2)

        if text_iccv_artifacts is not None:
            torch.save(text_iccv_artifacts.adapter_state, os.path.join(output_dir, "text_iccv_adapter.pt"))
            torch.save(text_iccv_artifacts.directions, os.path.join(output_dir, "text_iccv_directions.pt"))
            torch.save(
                text_iccv_artifacts.class_embeddings,
                os.path.join(output_dir, "text_iccv_class_embeddings.pt"),
            )
            write_iccv_json(
                os.path.join(output_dir, "text_iccv.json"),
                text_iccv_artifacts.to_json(),
            )

        if img_iccv_artifacts is not None:
            torch.save(img_iccv_artifacts.adapter_state, os.path.join(output_dir, "img_iccv_adapter.pt"))
            torch.save(
                img_iccv_artifacts.class_embeddings,
                os.path.join(output_dir, "img_iccv_class_embeddings.pt"),
            )
            write_iccv_json(
                os.path.join(output_dir, "img_iccv.json"),
                img_iccv_artifacts.to_json(),
            )
