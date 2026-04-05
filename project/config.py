from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model registry — shortcut keys for common backbones
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "bert":       "bert-base-uncased",
    "finbert":    "ProsusAI/finbert",
    "bertweet":   "vinai/bertweet-base",
    "roberta":    "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "mistral":    "mistralai/Mistral-7B-v0.1",
    "tinyllama":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen2":      "Qwen/Qwen2.5-0.5B",
}


def get_model_name(key: str) -> str:
    """Resolve a registry key or pass-through a full HuggingFace model ID."""
    return MODEL_REGISTRY.get(key, key)


def model_slug(key: str) -> str:
    """Return a filesystem-safe slug for cache directory naming."""
    return key.replace("/", "_").replace("-", "_")


@dataclass
class TransformerClassifierConfig:
    """Configuration for the reusable transformer fine-tuning path."""
    num_labels: int = 3
    max_length: int = 128
    batch_size: int = 32
    eval_batch_size: int = 64
    n_epochs: int = 5
    encoder_lr: float = 2e-5
    classifier_lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    unfreeze_layers: int = 4
    train_embeddings: bool = False
    pooling: str = "auto"   # "auto" | "cls" | "last" | "mean" | "pooler"
    input_mode: str = "text_plus_selected"  # "text" | "text_plus_selected" | "text_selected_pair"
    use_time_of_tweet: bool = False
    use_age_of_user: bool = False
    use_country: bool = False
    use_vader_features: bool = False
    use_afinn_features: bool = False
    loss_name: str = "cross_entropy"  # "cross_entropy" | "focal"
    focal_gamma: float = 1.5
    head_type: str = "mlp"  # "linear" | "mlp"
    hidden_dim: int = 0
    patience: int = 2
    label_smoothing: float = 0.0
    use_class_weights: bool = True
    grad_clip_norm: float = 1.0
    device: str = "cpu"
