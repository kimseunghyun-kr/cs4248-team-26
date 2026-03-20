"""
FinancialDirectionBank

Stores discovered concept directions for each axis.
Provides projection of CLS embeddings onto the bias subspace.
"""

import os
import torch
import torch.nn.functional as F
from typing import Dict, List
from tqdm import tqdm

from encoder import FinBERTEncoder
from direction_finder import PGDDirectionFinder
from config import PGDConfig


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------
AXIS_DEFS = [
    {
        "name": "bullish",
        "mode": "monopolar",
        "target_text": "stock surged earnings beat record revenue profits soared",
        "neutral_text": "company news market update",
    },
    {
        "name": "bearish",
        "mode": "monopolar",
        "target_text": "stock collapsed earnings missed revenue decline losses",
        "neutral_text": "company news market update",
    },
    {
        "name": "volatility",
        "mode": "monopolar",
        "target_text": "extreme volatility price swing turbulent market wild swings",
        "neutral_text": "market activity trading day",
    },
    {
        "name": "authority",
        "mode": "monopolar",
        "target_text": "CEO statement Fed announcement regulatory filing official report",
        "neutral_text": "tweet about finance",
    },
    {
        "name": "sentiment_polarity",
        "mode": "bipolar",
        "pos_text": "market booming great profits outstanding results",
        "neg_text": "market crashing losses disaster terrible results",
        "neutral_text": "market news",
    },
    {
        "name": "urgency",
        "mode": "bipolar",
        "pos_text": "breaking urgent immediate action required now",
        "neg_text": "routine stable no changes expected steady",
        "neutral_text": "routine update",
    },
]


class FinancialDirectionBank:
    """
    Holds K axes, each with N direction vectors.
    Projection onto each axis = mean |cosine similarity| with its directions.
    """

    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size
        # Dict[axis_name → Tensor(N, H)]
        self.directions: Dict[str, torch.Tensor] = {}
        self.axis_names: List[str] = []
        self.device = "cpu"

    def to(self, device: str):
        self.device = device
        self.directions = {k: v.to(device) for k, v in self.directions.items()}
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(
        self,
        encoder: FinBERTEncoder,
        anchor_texts: List[str],
        cfg: PGDConfig,
    ):
        """
        Discover directions for all axes using PGD.
        anchor_texts: first N texts from the training split.
        """
        finder = PGDDirectionFinder(encoder, cfg)

        for axis_def in tqdm(AXIS_DEFS, desc="Building direction bank"):
            name = axis_def["name"]
            mode = axis_def["mode"]
            print(f"  Discovering axis: {name} ({mode}) ...")

            kwargs = {k: v for k, v in axis_def.items() if k not in ("name", "mode")}
            directions = finder.collect_directions(
                mode=mode,
                anchor_texts=anchor_texts,
                n_directions=cfg.n_directions,
                **kwargs,
            )
            self.directions[name] = directions.cpu()
            self.axis_names.append(name)

        print(
            f"Direction Bank built: {len(self.axis_names)} axes "
            f"× {cfg.n_directions} directions each"
        )

    # ------------------------------------------------------------------
    # Project a batch of CLS embeddings onto the bank
    # ------------------------------------------------------------------
    def get_feature_vector(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: B × H (L2-normalized CLS embeddings)
        Returns: B × K  where K = number of axes
        Each feature = mean cosine similarity of z with the axis directions.
        """
        features = []
        for name in self.axis_names:
            dirs = self.directions[name].to(z.device)  # N × H
            # cosine sim: z (B×H) @ dirs.T (H×N) → B×N
            sims = z @ dirs.T                           # B × N
            feat = sims.mean(dim=-1, keepdim=True)      # B × 1
            features.append(feat)
        return torch.cat(features, dim=-1)              # B × K

    # ------------------------------------------------------------------
    # Convenience: project single axis
    # ------------------------------------------------------------------
    def project(self, z: torch.Tensor, axis_name: str) -> torch.Tensor:
        dirs = self.directions[axis_name].to(z.device)
        return (z @ dirs.T).mean(dim=-1)  # B

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "directions": self.directions,
                "axis_names": self.axis_names,
                "hidden_size": self.hidden_size,
            },
            path,
        )
        print(f"Direction bank saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FinancialDirectionBank":
        data = torch.load(path, map_location="cpu")
        bank = cls(hidden_size=data["hidden_size"])
        bank.directions = data["directions"]
        bank.axis_names = data["axis_names"]
        print(f"Direction bank loaded from {path}  ({len(bank.axis_names)} axes)")
        return bank

    def __len__(self):
        return len(self.axis_names)
