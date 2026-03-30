from dataclasses import dataclass

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
class CBDCConfig:
    """Configuration for the combined debias_vl + CBDC text_iccv pipeline.

    PGD inner-loop defaults match the original CBDC RN50 hyperparameters.
    """
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

    # debias_vl map
    n_bias_dirs: int = 4           # top-K SVD components from I - P_debias
    lambda_reg: float = 1000.0     # regularization: G = lambda * M + I

    device: str = "cpu"
