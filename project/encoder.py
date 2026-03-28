"""
Frozen FinBERT encoder with latent-space perturbation support.

Injects perturbation δ into the CLS token BEFORE the transformer layers by
using inputs_embeds — works for both BertModel (FinBERT) and DistilBertModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional


class FinBERTEncoder(nn.Module):
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name

        print(f"Loading tokenizer and model from '{model_name}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.to(device)
        self.hidden_size = self.backbone.config.hidden_size

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        print(f"  hidden_size={self.hidden_size} | all backbone params frozen")

    # ------------------------------------------------------------------
    # Internal: compute input embeddings (word + position + type)
    # ------------------------------------------------------------------
    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns token embedding matrix (B × L × H). No grad tracked on weights."""
        emb_module = self.backbone.embeddings
        return emb_module(input_ids=input_ids)

    # ------------------------------------------------------------------
    # Internal: backbone forward from inputs_embeds
    # ------------------------------------------------------------------
    def _forward_from_embeds(
        self,
        inputs_embeds: torch.Tensor,   # B × L × H
        attention_mask: torch.Tensor,  # B × L
    ) -> torch.Tensor:
        """
        Run backbone with pre-computed embeddings (supports both BERT & DistilBERT).
        Returns L2-normalized CLS embedding (B × H).
        Gradients flow through inputs_embeds, not through backbone weights.
        """
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # last_hidden_state: B × L × H  →  take CLS
        cls = out.last_hidden_state[:, 0, :]   # B × H
        return F.normalize(cls, dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tokenize(self, texts, max_length: int = 128):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode_text(self, texts, max_length: int = 128) -> torch.Tensor:
        """Encode a list of strings → normalized CLS vectors (B × H)."""
        enc = self.tokenize(texts, max_length)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        embeds = self._get_embeddings(input_ids)
        return self._forward_from_embeds(embeds, attention_mask)

    def encode_with_delta(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        delta: torch.Tensor,           # B × H  (perturbation for CLS token only)
    ) -> torch.Tensor:
        """
        Differentiable encode with δ injected at CLS token embedding.
        Gradients flow to delta through the frozen backbone (backbone weights
        have requires_grad=False so they don't accumulate gradients).
        """
        # Compute base embeddings (in-graph since delta needs grad)
        embeds = self._get_embeddings(input_ids)   # B × L × H

        # Build a full-length perturbation tensor: non-zero only at CLS (pos 0).
        # Use mask multiplication to avoid in-place ops that can break autograd.
        B, L, H = embeds.shape
        # mask: (B, L, H) with 1s only at position 0
        mask = torch.zeros(B, L, H, device=self.device)
        mask[:, 0, :] = 1.0
        # Broadcast delta (B×H) → (B×L×H) zeroed outside CLS
        delta_full = delta.unsqueeze(1).expand(B, L, H) * mask   # B × L × H

        perturbed = embeds + delta_full

        return self._forward_from_embeds(perturbed, attention_mask)

    def get_intermediate_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_layers: int = None,
    ) -> torch.Tensor:
        """
        Run embedding layer + first n_layers transformer layers, return hidden states.

        NLP analog of CLIP RN50's get_feature(): the frozen encoder body that produces
        the intermediate representation z before the last-layer tail.

        n_layers defaults to num_hidden_layers - 1 (all but the last layer).
        Call under torch.no_grad() — this is the frozen body; no grad needed here.

        Returns: (B, L, H) hidden states at the n_layers-th layer boundary.
        """
        if n_layers is None:
            n_layers = self.backbone.config.num_hidden_layers - 1
        hidden = self._get_embeddings(input_ids)
        for layer_module in self.backbone.encoder.layer[:n_layers]:
            hidden = layer_module(hidden, attention_mask)[0]
        return hidden  # (B, L, H)

    def encode_with_delta_from_hidden(
        self,
        hidden_states: torch.Tensor,   # (B, L, H)  detached intermediate repr
        attention_mask: torch.Tensor,  # (B, L)
        delta: torch.Tensor,           # (B, H)  perturbation for CLS token only
        start_layer: int = None,
    ) -> torch.Tensor:
        """
        NLP analog of CLIP RN50's target_model.forward(z_adv): inject δ at the CLS
        position of an intermediate hidden state, then run the last transformer layer(s).

        hidden_states should be detached (produced by get_intermediate_features under
        no_grad). delta carries the gradient: loss → layer[start_layer:] → delta.

        start_layer defaults to num_hidden_layers - 1 (only the last layer runs).

        Returns: L2-normalized CLS embedding (B, H).
        """
        if start_layer is None:
            start_layer = self.backbone.config.num_hidden_layers - 1
        B, L, H = hidden_states.shape
        mask = torch.zeros(B, L, H, device=delta.device)
        mask[:, 0, :] = 1.0
        delta_full = delta.unsqueeze(1).expand(B, L, H) * mask   # (B, L, H)
        perturbed = hidden_states + delta_full  # grad flows through delta_full → delta
        for layer_module in self.backbone.encoder.layer[start_layer:]:
            perturbed = layer_module(perturbed, attention_mask)[0]
        cls = perturbed[:, 0, :]
        return F.normalize(cls, dim=-1)

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pre-tokenized ids without perturbation."""
        embeds = self._get_embeddings(input_ids)
        return self._forward_from_embeds(embeds, attention_mask)
