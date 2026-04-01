from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model registry — shortcut keys for common BERT-derivative backbones
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "bert":       "bert-base-uncased",
    "finbert":    "ProsusAI/finbert",
    "bertweet":   "vinai/bertweet-base",
    "roberta":    "roberta-base",
    "distilbert": "distilbert-base-uncased",
}


def get_model_name(key: str) -> str:
    """Resolve a registry key or pass-through a full HuggingFace model ID."""
    return MODEL_REGISTRY.get(key, key)


def model_slug(key: str) -> str:
    """Return a filesystem-safe slug for cache directory naming."""
    return key.replace("/", "_").replace("-", "_")


@dataclass
class CBDCConfig:
    """Configuration for the combined debias_vl + CBDC text_iccv pipeline.

    PGD inner-loop defaults match the original CBDC RN50 hyperparameters.
    Works with any BERT-derivative backbone.
    """
    # Prompt wording — controls the text unit in generated prompts
    text_unit: str = "text"        # "text", "tweet", "review", "post", etc.

    # PGD inner loop (matches RN50 / CelebA defaults)
    epsilon: float = 1.0           # L-inf perturbation bound (att_bnd)
    n_pgd_steps: int = 20          # sign-SGD iterations per restart (att_itr)
    step_lr: float = 0.0037        # sign-SGD step size (att_stp)
    keep_weight: float = 0.92      # L_s weight inside PGD (keep_weight)
    num_samples: int = 10          # PGD restarts per epoch (num_sam)
    random_eps: float = 0.22       # random init radius (rand)

    # text_iccv training loop
    n_epochs: int = 100            # encoder training epochs (txt_iters)
    lr: float = 1e-5               # AdamW lr for layer 11
    up_scale: float = 100.0        # loss multiplier (up_)
    eval_every: int = 10           # validation selector frequency during Phase 2
    selector_train_per_class: int = 512   # balanced train subset size per class
    selector_batch_size: int = 128        # batch size for selector re-encoding

    # debias_vl topic mining / pole instantiation
    use_mined_topics: bool = True
    mine_max_topics: int = 32
    mine_min_doc_freq: int = 20
    mine_max_doc_freq_ratio: float = 0.20
    pole_phrases_per_side: int = 4

    # debias_vl map
    n_bias_dirs: int = 4           # top-K SVD components from I - P_debias
    lambda_reg: float = 1000.0     # regularization: G = lambda * M + I

    # Sentiment-orthogonal PGD constraint (Strategy A)
    # Projects PGD gradient orthogonal to sentiment prototypes at each step,
    # preventing confound directions from drifting into sentiment space.
    sent_orthogonal_pgd: bool = True

    device: str = "cpu"


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
    pooling: str = "cls"   # "cls" | "mean" | "pooler"
    input_mode: str = "text_plus_selected"  # "text" | "text_plus_selected" | "text_selected_pair"
    use_time_of_tweet: bool = False
    use_age_of_user: bool = False
    use_country: bool = False
    loss_name: str = "cross_entropy"  # "cross_entropy" | "focal"
    focal_gamma: float = 1.5
    head_type: str = "mlp"  # "linear" | "mlp"
    hidden_dim: int = 0
    patience: int = 2
    label_smoothing: float = 0.0
    use_class_weights: bool = True
    grad_clip_norm: float = 1.0
    device: str = "cpu"
