from dataclasses import dataclass, field
from typing import List


@dataclass
class PGDConfig:
    epsilon: float = 0.08
    n_steps: int = 15
    step_lr: float = 0.008       # Adam lr for delta optimizer
    lambda_s: float = 0.4        # weight of semantic preservation loss
    n_directions: int = 8        # directions collected per axis
    n_anchors: int = 100         # anchor tweets drawn from train split
    device: str = "cpu"          # overridden at runtime


@dataclass
class TrainConfig:
    model_name: str = "ProsusAI/finbert"
    fallback_model_name: str = "distilbert-base-uncased"
    hidden_size: int = 768       # FinBERT hidden dim; distilbert is also 768
    n_concept_axes: int = 6
    n_directions_per_axis: int = 8

    mlp_hidden: int = 256
    mlp_dropout: float = 0.1

    epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10

    seed: int = 42
    device: str = "cpu"          # overridden at runtime

    direction_bank_path: str = "direction_bank.pt"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    # Dataset
    dataset_name: str = "tweet_eval"
    dataset_config: str = "sentiment"
