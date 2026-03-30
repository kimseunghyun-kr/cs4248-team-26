"""
Frozen encoder with latent-space perturbation support.

Architecture correspondence (RN50 ↔ FinBERT):
  CLIP RN50:  frozen ResNet body → perturb intermediate → attnpool + c_proj (1 MHA + 1 linear)
  FinBERT:    frozen layers 0–10 → perturb h_layer10    → layer 11 (1 transformer layer)

Both tails are single-attention-layer structures. Gradients flow through
the tail to delta; backbone weights remain frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


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

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        print(f"  hidden_size={self.hidden_size} | all backbone params frozen")

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def tokenize(self, texts, max_length: int = 128):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    # ------------------------------------------------------------------
    # Encode text strings → normalized CLS vectors
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, texts, max_length: int = 128) -> torch.Tensor:
        """Encode a list of strings → normalized CLS vectors (B × H)."""
        enc = self.tokenize(texts, max_length)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, dim=-1)

    # ------------------------------------------------------------------
    # Encode pre-tokenized ids (no perturbation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pre-tokenized ids → normalized CLS vectors (B × H)."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, dim=-1)

    # ------------------------------------------------------------------
    # Get intermediate features (frozen body, layers 0 to n_layers-1)
    # ------------------------------------------------------------------
    def get_intermediate_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_layers: int = None,
    ) -> torch.Tensor:
        """
        Run frozen body (layers 0 to n_layers-1), return hidden states.
        Default n_layers = num_hidden_layers - 1 (all but last layer).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if n_layers is None:
            n_layers = self.backbone.config.num_hidden_layers - 1

        with torch.no_grad():
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return out.hidden_states[n_layers]

    # ------------------------------------------------------------------
    # SDPA mask helper
    # ------------------------------------------------------------------
    def _prepare_sdpa_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Create contiguous 4D additive float mask (B, 1, L, L)."""
        B, L = attention_mask.shape
        mask = torch.zeros(B, 1, L, L, dtype=dtype, device=attention_mask.device)
        mask = mask.masked_fill(attention_mask[:, None, None, :] == 0, torch.finfo(dtype).min)
        if not mask.is_contiguous():
            mask = mask.contiguous()
        return mask

    # ------------------------------------------------------------------
    # Encode with delta perturbation at CLS position (through layer 11)
    # ------------------------------------------------------------------
    def encode_with_delta_from_hidden(
        self,
        hidden_states: torch.Tensor,   # (B, L, H) detached intermediate repr
        attention_mask: torch.Tensor,   # (B, L)
        delta: torch.Tensor,           # (B, H) perturbation for CLS token only
        start_layer: int = None,
    ) -> torch.Tensor:
        """
        NLP analog of CLIP RN50's target_model.forward(z_adv).

        Injects delta at the CLS token position of hidden_states, then
        runs through layer 11 (the trainable tail). Gradients flow to
        delta and through layer 11 weights.
        """
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if delta.dim() == 1:
            delta = delta.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        B, L, H = hidden_states.shape

        # Ensure attention_mask batch size matches
        if attention_mask.size(0) != B:
            attention_mask = attention_mask[:B]

        if start_layer is None:
            start_layer = self.backbone.config.num_hidden_layers - 1

        # 4D attention mask for SDPA
        sdpa_mask = self._prepare_sdpa_mask(attention_mask, hidden_states.dtype)

        # Inject delta at CLS position (index 0) only
        mask = torch.zeros(B, L, H, device=delta.device)
        mask[:, 0, :] = 1.0
        delta_full = delta.unsqueeze(1).expand(B, L, H) * mask
        perturbed = hidden_states + delta_full

        # Forward through tail layer(s)
        for layer_module in self.backbone.encoder.layer[start_layer:]:
            out = layer_module(hidden_states=perturbed, attention_mask=sdpa_mask)
            perturbed = out[0] if isinstance(out, tuple) else out

        cls = perturbed[:, 0, :]
        return F.normalize(cls, dim=-1)
