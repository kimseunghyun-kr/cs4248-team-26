"""
Frozen encoder with latent-space perturbation support.

Architecture correspondence (RN50 ↔ BERT-derivative):
  CLIP RN50:       frozen ResNet body → perturb intermediate → attnpool + c_proj (1 MHA + 1 linear)
  BERT-derivative: frozen layers 0–N  → perturb h_layer_N   → layer N+1 (trainable tail)

Both tails are single-attention-layer structures. Gradients flow through
the tail to delta; backbone weights remain frozen.

Phase 1 embedding extraction works with many HuggingFace backbones.
Phase 2 latent-tail CBDC training supports:
  - encoder-style bidirectional models with CLS-style pooling
  - decoder-only Llama/Qwen2/Gemma4-style models via a terminal-token tail path
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


SUPPORTED_LATENT_TAIL_MODEL_TYPES = {
    "bert",
    "roberta",
    "xlm-roberta",
    "distilbert",
    "llama",
    "qwen2",
    "gemma4",
}


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
        self.load_dtype = self._resolve_load_dtype(model_name)
        if tokenizer is not None:
            self.tokenizer = tokenizer
            print(f"  Using custom tokenizer: {type(tokenizer).__name__}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model_name and self._looks_decoder_model_name(model_name):
            self.tokenizer.padding_side = "right"
        try:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                dtype=self.load_dtype,
            )
        except ValueError as exc:
            if "model type `gemma4`" in str(exc):
                raise RuntimeError(
                    "Gemma 4 checkpoints require a Transformers build that recognizes "
                    "`model_type=gemma4`. The current environment can tokenize Gemma 4, "
                    "but cannot load the model body yet. Upgrade Transformers on the H100 "
                    "node first, then rerun Phase 1/2."
                ) from exc
            raise
        self.backbone.to(device)
        self.core_model = self._resolve_core_model(self.backbone)
        if self.load_dtype == torch.float32:
            self.backbone.float()
        else:
            self.backbone.to(dtype=self.load_dtype)
        self.model_type = getattr(self.backbone.config, "model_type", "unknown")
        self.hidden_size = self._core_config().hidden_size
        self.backbone_dtype = next(self.core_model.parameters()).dtype
        self._validate_gemma_variant_support()

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        print(
            f"  model_type={self.model_type} | hidden_size={self.hidden_size} "
            f"| all backbone params frozen | dtype={self.backbone_dtype}"
        )

    @staticmethod
    def _looks_decoder_model_name(model_name: str) -> bool:
        lower = model_name.lower()
        return "llama" in lower or "qwen" in lower or "gemma-4" in lower or "gemma4" in lower

    @staticmethod
    def _resolve_load_dtype(model_name: str) -> torch.dtype:
        dtype_name = os.environ.get("MODEL_DTYPE", "").strip().lower()
        if dtype_name:
            dtype_map = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            if dtype_name not in dtype_map:
                raise ValueError(
                    f"Unsupported MODEL_DTYPE='{dtype_name}'. "
                    f"Expected one of {sorted(dtype_map)}."
                )
            return dtype_map[dtype_name]

        lower = model_name.lower()
        if "gemma-4" in lower or "gemma4" in lower:
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _resolve_core_model(backbone: nn.Module) -> nn.Module:
        if getattr(backbone.config, "model_type", None) == "gemma4" and hasattr(backbone, "language_model"):
            return backbone.language_model
        return backbone

    def _core_config(self):
        return self.core_model.config

    def _validate_gemma_variant_support(self) -> None:
        if self.model_type != "gemma4":
            return
        cfg = self._core_config()
        uses_per_layer_inputs = bool(getattr(cfg, "hidden_size_per_layer_input", 0))
        uses_shared_kv = bool(getattr(cfg, "num_kv_shared_layers", 0))
        if uses_per_layer_inputs or uses_shared_kv:
            raise NotImplementedError(
                "Current Gemma 4 support is intentionally scoped to 26B/31B-style text variants "
                "without per-layer inputs or shared-KV text layers. "
                "The smaller E2B/E4B variants are not supported in this cleaned-up path."
            )

    def is_decoder_family(self) -> bool:
        return self.model_type in {"llama", "qwen2", "gemma4"}

    def supports_latent_tail(self) -> bool:
        return self.model_type in SUPPORTED_LATENT_TAIL_MODEL_TYPES

    def _require_latent_tail_support(self) -> None:
        if self.supports_latent_tail():
            return
        raise NotImplementedError(
            f"Phase 2 latent-tail CBDC is not yet implemented for model_type='{self.model_type}' "
            f"({self.model_name}). Supported families: "
            f"{sorted(SUPPORTED_LATENT_TAIL_MODEL_TYPES)}. "
            "Phase 1 embedding extraction may still work, but Phase 2 needs an architecture-specific tail path."
        )

    # ------------------------------------------------------------------
    # Transformer layer resolution (BERT vs DistilBERT vs others)
    # ------------------------------------------------------------------
    def _get_transformer_layers(self):
        """Resolve the sequential list of transformer layers.

        Handles architecture differences:
          - BERT / RoBERTa / FinBERT / DeBERTa: backbone.encoder.layer
          - DistilBERT:                          backbone.transformer.layer
          - LLaMA / Qwen2 / Gemma4-text:         backbone.layers
        """
        self._require_latent_tail_support()
        if hasattr(self.core_model, "layers"):
            return self.core_model.layers
        if hasattr(self.core_model, "encoder") and hasattr(self.core_model.encoder, "layer"):
            return self.core_model.encoder.layer
        if hasattr(self.core_model, "transformer") and hasattr(self.core_model.transformer, "layer"):
            return self.core_model.transformer.layer
        raise AttributeError(
            f"Cannot find transformer layers in {type(self.core_model).__name__}. "
            f"Expected .encoder.layer or .transformer.layer for a supported encoder model."
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

    def _sequence_positions(self, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.long().sum(dim=-1).clamp_min(1)
        return lengths - 1

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_decoder_family():
            positions = self._sequence_positions(attention_mask)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, positions, :]
        return hidden_states[:, 0, :]

    def _injection_positions(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.is_decoder_family():
            return self._sequence_positions(attention_mask)
        return torch.zeros(attention_mask.size(0), dtype=torch.long, device=attention_mask.device)

    def _num_hidden_layers(self) -> int:
        cfg = self._core_config()
        num_layers = getattr(cfg, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(cfg, "n_layers", None)
        if num_layers is None:
            raise AttributeError(
                f"Cannot determine hidden-layer count for model_type='{self.model_type}' ({self.model_name})."
            )
        return int(num_layers)

    def _prepare_decoder_tail_inputs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object, dict[str, torch.Tensor], dict]:
        seq_len = hidden_states.shape[1]
        cache_position = torch.arange(seq_len, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": self._core_config(),
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        full_attention = create_causal_mask(**mask_kwargs)
        attention_map = {"full_attention": full_attention}
        has_sliding_layers = getattr(self.core_model, "has_sliding_layers", False)
        sliding_window = getattr(self._core_config(), "sliding_window", None)
        if self.model_type in {"qwen2", "gemma4"} and (has_sliding_layers or sliding_window):
            attention_map["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        decoder_aux: dict = {}
        if self.model_type == "gemma4":
            layer_types = list(self._core_config().layer_types)
            unique_layer_types = set(layer_types)
            position_embeddings = {
                layer_type: self.core_model.rotary_emb(hidden_states, position_ids, layer_type)
                for layer_type in unique_layer_types
            }
            decoder_aux["layer_types"] = layer_types
            decoder_aux["shared_kv_states"] = {}
        else:
            position_embeddings = self.core_model.rotary_emb(hidden_states, position_ids)

        return cache_position, position_ids, position_embeddings, attention_map, decoder_aux

    # ------------------------------------------------------------------
    # Encode text strings → normalized CLS vectors
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, texts, max_length: int = 128) -> torch.Tensor:
        """Encode a list of strings → normalized sentence vectors (B × H)."""
        enc = self.tokenize(texts, max_length)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        out = self.core_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_hidden_states(out.last_hidden_state, attention_mask)
        return F.normalize(pooled.float(), dim=-1)

    # ------------------------------------------------------------------
    # Encode pre-tokenized ids (no perturbation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pre-tokenized ids → normalized sentence vectors (B × H)."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        out = self.core_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_hidden_states(out.last_hidden_state, attention_mask)
        return F.normalize(pooled.float(), dim=-1)

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
        self._require_latent_tail_support()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if n_layers is None:
            n_layers = self._num_hidden_layers() - 1

        with torch.no_grad():
            out = self.core_model(
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
        self._require_latent_tail_support()

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if delta.dim() == 1:
            delta = delta.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        hidden_states = hidden_states.to(self.device, dtype=self.backbone_dtype)
        attention_mask = attention_mask.to(self.device)
        delta = delta.to(self.device, dtype=self.backbone_dtype)

        B, L, H = hidden_states.shape

        # Ensure attention_mask batch size matches
        if attention_mask.size(0) != B:
            attention_mask = attention_mask[:B]

        if start_layer is None:
            start_layer = self._num_hidden_layers() - 1

        inject_positions = self._injection_positions(attention_mask)

        mask = torch.zeros(B, L, H, device=delta.device, dtype=delta.dtype)
        batch_idx = torch.arange(B, device=delta.device)
        mask[batch_idx, inject_positions, :] = 1.0
        delta_full = delta.unsqueeze(1).expand(B, L, H) * mask
        perturbed = hidden_states + delta_full

        if self.is_decoder_family():
            cache_position, position_ids, position_embeddings, attention_map, decoder_aux = self._prepare_decoder_tail_inputs(
                perturbed,
                attention_mask,
            )
            for layer_idx, layer_module in enumerate(self._get_transformer_layers()[start_layer:], start=start_layer):
                if self.model_type == "gemma4":
                    layer_type = decoder_aux["layer_types"][layer_idx]
                    perturbed = layer_module(
                        perturbed,
                        shared_kv_states=decoder_aux["shared_kv_states"],
                        position_embeddings=position_embeddings[layer_type],
                        attention_mask=attention_map[layer_type],
                        position_ids=position_ids,
                        past_key_values=None,
                    )
                    continue
                attn_key = getattr(layer_module, "attention_type", None)
                if attn_key in attention_map:
                    attention_mask_layer = attention_map[attn_key]
                else:
                    attention_mask_layer = attention_map["full_attention"]
                perturbed = layer_module(
                    perturbed,
                    attention_mask=attention_mask_layer,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                )
            perturbed = self.core_model.norm(perturbed)
            pooled = self._pool_hidden_states(perturbed, attention_mask)
            return F.normalize(pooled.float(), dim=-1)

        # 4D attention mask for SDPA
        sdpa_mask = self._prepare_sdpa_mask(attention_mask, hidden_states.dtype)

        # Forward through tail layer(s)
        for layer_module in self._get_transformer_layers()[start_layer:]:
            out = layer_module(hidden_states=perturbed, attention_mask=sdpa_mask)
            perturbed = out[0] if isinstance(out, tuple) else out

        pooled = self._pool_hidden_states(perturbed, attention_mask)
        return F.normalize(pooled.float(), dim=-1)
