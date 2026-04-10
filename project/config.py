# ---------------------------------------------------------------------------
# Model registry — shortcut keys for reusable transformer backbones
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "bert":       "bert-base-cased",
    "bert-uncased": "google-bert/bert-base-uncased",
    "bert-base-uncased": "google-bert/bert-base-uncased",
    "bert-large-uncased": "google-bert/bert-large-uncased",
    "finbert":    "ProsusAI/finbert",
    "bertweet":   "vinai/bertweet-base",
    "roberta":    "roberta-base",
    "roberta-large": "FacebookAI/roberta-large",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "qwen25-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen25-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen25-3b": "Qwen/Qwen2.5-3B",
    "qwen25-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "llama32-1b": "meta-llama/Llama-3.2-1B",
    "llama32-3b": "meta-llama/Llama-3.2-3B",
    "gemma4-26b": "google/gemma-4-26B-A4B",
    "gemma4-26b-it": "google/gemma-4-26B-A4B-it",
    "distilbert": "distilbert-base-cased",
}


def get_model_name(key: str) -> str:
    """Resolve a registry key or pass-through a full HuggingFace model ID."""
    return MODEL_REGISTRY.get(key, key)


def model_slug(key: str) -> str:
    """Return a filesystem-safe slug for cache directory naming."""
    return key.replace("/", "_").replace("-", "_")
