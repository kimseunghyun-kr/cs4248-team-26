"""
CBDCSentimentClassifier

Architecture:
  frozen FinBERT → CLS z (B×768)
  direction bank projection → proj_features (B×K)
  concat([z, proj_features]) → B×(768+K)
  MLP head → logits (B×3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import FinBERTEncoder
from direction_bank import FinancialDirectionBank
from config import TrainConfig


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CBDCSentimentClassifier(nn.Module):
    def __init__(
        self,
        encoder: FinBERTEncoder,
        bank: FinancialDirectionBank,
        cfg: TrainConfig,
        ablation: str = "full",   # "full" | "z_only" | "directions_only"
    ):
        super().__init__()
        self.encoder = encoder
        self.bank = bank
        self.ablation = ablation
        K = len(bank)

        if ablation == "full":
            in_dim = cfg.hidden_size + K
        elif ablation == "z_only":
            in_dim = cfg.hidden_size
        elif ablation == "directions_only":
            in_dim = K
        else:
            raise ValueError(f"Unknown ablation mode: {ablation}")

        self.head = MLPHead(
            in_dim=in_dim,
            hidden=cfg.mlp_hidden,
            n_classes=3,
            dropout=cfg.mlp_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # CLS embedding from frozen encoder
        with torch.no_grad():
            z = self.encoder.encode_ids(input_ids, attention_mask)  # B × H

        if self.ablation == "z_only":
            features = z
        elif self.ablation == "directions_only":
            features = self.bank.get_feature_vector(z)              # B × K
        else:  # full
            proj = self.bank.get_feature_vector(z)                  # B × K
            features = torch.cat([z, proj], dim=-1)                 # B × (H+K)

        return self.head(features)                                   # B × 3
