"""
Frozen encoder with latent-space perturbation support.

Architecture correspondence (RN50 ↔ BERT-derivative):
  CLIP RN50:       frozen ResNet body → perturb intermediate → attnpool + c_proj (1 MHA + 1 linear)
  BERT-derivative: frozen layers 0–N  → perturb h_layer_N   → layer N+1 (trainable tail)

Both tails are single-attention-layer structures. Gradients flow through
the tail to delta; backbone weights remain frozen.

Works with any HuggingFace AutoModel that exposes transformer layers
(BERT, RoBERTa, FinBERT, BERTweet, DistilBERT, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
        tokenizer=None,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name

        print(f"Loading model from '{model_name}' ...")
        if tokenizer is not None:
            self.tokenizer = tokenizer
            print(f"  Using custom tokenizer: {type(tokenizer).__name__}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            fallback_pad = (
                self.tokenizer.eos_token
                or self.tokenizer.sep_token
                or self.tokenizer.unk_token
            )
            if fallback_pad is not None:
                self.tokenizer.pad_token = fallback_pad
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.to(device)
        self.hidden_size = self.backbone.config.hidden_size
        if getattr(self.backbone.config, "pad_token_id", None) is None:
            self.backbone.config.pad_token_id = self.tokenizer.pad_token_id

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        print(f"  hidden_size={self.hidden_size} | all backbone params frozen")

    # ------------------------------------------------------------------
    # Transformer layer resolution (BERT vs DistilBERT vs others)
    # ------------------------------------------------------------------
    def _get_transformer_layers(self):
        """Resolve the sequential list of transformer layers.

        Handles architecture differences:
          - BERT / RoBERTa / FinBERT / DeBERTa: backbone.encoder.layer
          - DistilBERT:                          backbone.transformer.layer
          - GPT-2 / GPT-Neo:                     backbone.transformer.h
          - LLaMA / Qwen2 / Mistral:             backbone.model.layers
          - CLIP text encoder:                    backbone.text_model.encoder.layers
        Note: Causal models (GPT-2, LLaMA, Qwen2) need last-token pooling and
        delta injection at the last non-pad position rather than CLS at pos 0.
        """
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        if hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "layer"):
            return self.backbone.transformer.layer
        # GPT-2 / GPT-Neo
        if hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "h"):
            return self.backbone.transformer.h
        # LLaMA / Qwen2 / Mistral
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            return self.backbone.model.layers
        # CLIP text encoder
        if hasattr(self.backbone, "text_model") and hasattr(self.backbone.text_model, "encoder"):
            return self.backbone.text_model.encoder.layers
        raise AttributeError(
            f"Cannot find transformer layers in {type(self.backbone).__name__}. "
            f"Expected .encoder.layer, .transformer.layer, .transformer.h, "
            f".model.layers, or .text_model.encoder.layers"
        )

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

    def _pool_output(self, outputs, attention_mask: torch.Tensor, pooling: str = "cls") -> torch.Tensor:
        if pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            return (outputs.last_hidden_state * mask).sum(dim=1) / denom
        if pooling != "cls":
            raise ValueError(f"Unsupported pooling='{pooling}'. Use 'cls', 'mean', or 'pooler'.")
        return outputs.last_hidden_state[:, 0, :]

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str = "cls",
        normalize: bool = False,
    ) -> torch.Tensor:
        """Forward through the backbone with gradient enabled."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_output(outputs, attention_mask, pooling=pooling)
        return F.normalize(pooled, dim=-1) if normalize else pooled

    def set_trainable_layers(self, n_layers: int = 0, train_embeddings: bool = False) -> None:
        """Freeze the backbone, then unfreeze the last `n_layers` transformer blocks."""
        if n_layers < 0:
            raise ValueError("n_layers must be >= 0")

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if train_embeddings and hasattr(self.backbone, "embeddings"):
            for p in self.backbone.embeddings.parameters():
                p.requires_grad_(True)

        layers = list(self._get_transformer_layers())
        n_layers = min(n_layers, len(layers))
        if n_layers > 0:
            for layer_module in layers[-n_layers:]:
                for p in layer_module.parameters():
                    p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"  trainable backbone params={trainable:,} | unfrozen_layers={n_layers} | "
              f"train_embeddings={train_embeddings}")

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
    # Encode with delta perturbation at CLS position (through tail layer)
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
        runs through the trainable tail layer(s). Gradients flow to
        delta and through the tail weights.
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
        for layer_module in self._get_transformer_layers()[start_layer:]:
            out = layer_module(hidden_states=perturbed, attention_mask=sdpa_mask)
            perturbed = out[0] if isinstance(out, tuple) else out

        cls = perturbed[:, 0, :]
        return F.normalize(cls, dim=-1)
