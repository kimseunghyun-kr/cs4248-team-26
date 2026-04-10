from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os

import torch

from config import get_model_name, model_slug
from encoder import TransformerEncoder
from .data import ConceptMediaRecord, build_dataloaders, write_manifest
from .pgd import DiscoveryArtifacts, discover_nuisance_directions
from .prompts import PersonalizationPromptBank, build_prompt_bank


@dataclass
class PersonalizationConfig:
    model: str = "qwen25-3b"
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
    device: str = "cpu"


class PersonalizationTrainer:
    def __init__(self, cfg: PersonalizationConfig):
        self.cfg = cfg
        self.model_name = get_model_name(cfg.model)
        self.cache_slug = model_slug(cfg.model)
        tokenizer = None
        if cfg.tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        self.encoder = TransformerEncoder(
            model_name=self.model_name,
            device=cfg.device,
            tokenizer=tokenizer,
        )

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

    def save_run(
        self,
        output_dir: str,
        prompt_bank: PersonalizationPromptBank,
        data_payload: dict,
        discovery: DiscoveryArtifacts | None = None,
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

        if discovery is None:
            return

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
