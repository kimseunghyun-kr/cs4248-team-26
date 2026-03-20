from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Model registry for ablation studies
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "finbert":  "ProsusAI/finbert",
    "bert":     "bert-base-uncased",
    "bertweet": "vinai/bertweet-base",
}


def get_model_name(key: str) -> str:
    """Resolve a registry key or pass-through a full HuggingFace model ID."""
    return MODEL_REGISTRY.get(key, key)


def model_slug(key: str) -> str:
    """Return a filesystem-safe slug for cache directory naming."""
    return key.replace("/", "_").replace("-", "_")


@dataclass
class PGDConfig:
    epsilon: float = 0.10        # perturbation ball radius (increased from 0.08)
    n_steps: int = 50            # Adam steps (increased from 15; needs ~50 to converge)
    step_lr: float = 0.01        # Adam lr for delta optimizer
    lambda_s: float = 0.2        # semantic preservation weight (loosened from 0.4)
    n_directions: int = 16       # directions collected (increased from 8)
    n_anchors: int = 500         # anchor samples for refinement (increased from 100)
    device: str = "cpu"          # overridden at runtime


@dataclass
class SAEConfig:
    hidden_dim: int = 1536       # 2× overcomplete dictionary (768 × 2)
    lambda_l1: float = 1e-3      # sparsity penalty (tune: 1e-4 if too sparse, 5e-3 if too dense)
    lr: float = 1e-4
    epochs: int = 50
    batch_size: int = 256
    top_k: int = 32              # top-K style features used to build v_style
    checkpoint_path: str = "cache/sae_checkpoint.pt"
    v_style_path: str = "cache/v_style.pt"


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
