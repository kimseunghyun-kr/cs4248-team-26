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

# ------------------------------------------------------------------
    # SDPA Mask Helper
    # ------------------------------------------------------------------
    def _prepare_sdpa_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Creates a contiguous 4D additive float mask (B, 1, L, L) to completely bypass
        PyTorch SDPA C++ broadcasting and zero-stride memory layout bugs.
        """
        if attention_mask.dim() != 2:
            raise ValueError(f"[Mask Error] Expected 2D attention_mask (B, L), got {attention_mask.shape}")
            
        B, L = attention_mask.shape
        # Allocate real memory for the 4D mask (0.0 means attend)
        mask = torch.zeros(B, 1, L, L, dtype=dtype, device=attention_mask.device)
        # Fill padding tokens with a large negative value (-inf equivalent)
        mask = mask.masked_fill(attention_mask[:, None, None, :] == 0, torch.finfo(dtype).min)
        
        # Defensive check to ensure memory is contiguous before passing to C++
        if not mask.is_contiguous():
            mask = mask.contiguous()
            
        return mask

# ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------
    def get_intermediate_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_layers: int = None,
    ) -> torch.Tensor:
        """
        Run embedding layer + first n_layers transformer layers, return hidden states.
        """
        # 1. Fix the 1D tensor bug from refine.py (input_ids was passed as (128,))
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Make it (1, 128)

        # 2. Rebuild the true mask because refine.py passes the (500, 128) anchor mask
        pad_token_id = self.tokenizer.pad_token_id or 0
        real_attention_mask = (input_ids != pad_token_id).long()
        
        # 3. Cache the true mask so the hot loop can use it later
        self._cached_tweet_mask = real_attention_mask

        if n_layers is None:
            n_layers = self.backbone.config.num_hidden_layers - 1

        # Let the native backbone handle the frozen body to avoid all SDPA bugs
        with torch.no_grad():
            embeds = self._get_embeddings(input_ids)
            outputs = self.backbone(
                inputs_embeds=embeds,
                attention_mask=real_attention_mask,
                output_hidden_states=True
            )
            # outputs.hidden_states[11] is exactly the output of layer 10
            return outputs.hidden_states[n_layers]

    def encode_with_delta_from_hidden(
        self,
        hidden_states: torch.Tensor,   # (B, L, H)  detached intermediate repr
        attention_mask: torch.Tensor,  # (B, L)
        delta: torch.Tensor,           # (B, H)  perturbation for CLS token only
        start_layer: int = None,
    ) -> torch.Tensor:
        """
        NLP analog of CLIP RN50's target_model.forward(z_adv).
        """
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if delta.dim() == 1:
            delta = delta.unsqueeze(0)

        B, L, H = hidden_states.shape

        # 1. Use the cached true mask to bypass refine.py's anchor mask
        if hasattr(self, "_cached_tweet_mask") and self._cached_tweet_mask.size(0) == B:
            attention_mask = self._cached_tweet_mask
        else:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.size(0) != B:
                attention_mask = attention_mask[:B]

        if start_layer is None:
            start_layer = self.backbone.config.num_hidden_layers - 1

        # 2. Use our foolproof contiguous float mask
        sdpa_mask = self._prepare_sdpa_mask(attention_mask, hidden_states.dtype)

        mask = torch.zeros(B, L, H, device=delta.device)
        mask[:, 0, :] = 1.0
        delta_full = delta.unsqueeze(1).expand(B, L, H) * mask   # (B, L, H)
        perturbed = hidden_states + delta_full  
        
        # 3. Robust layer loop
        for i, layer_module in enumerate(self.backbone.encoder.layer[start_layer:]):
            try:
                # Pass as kwargs to prevent positional API mismatch
                out = layer_module(hidden_states=perturbed, attention_mask=sdpa_mask)
                
                # FIX: Handle both tuple and raw tensor returns safely
                perturbed = out[0] if isinstance(out, tuple) else out
                
            except Exception as e:
                actual_layer_idx = start_layer + i
                print(f"\n[FATAL ERROR] encode_with_delta_from_hidden crashed at layer {actual_layer_idx}!")
                raise e
            
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
