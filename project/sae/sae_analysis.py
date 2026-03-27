"""
Phase 3a: Extract tweet-style direction v_style from trained SAE.

For each SAE feature i (0..hidden_dim-1):
    tweet_mean_act_i  = mean activation on z_tweet_train
    formal_mean_act_i = mean activation on z_formal
    style_score_i     = tweet_mean_act_i - formal_mean_act_i

top-K features = argsort(style_score)[-K:]
v_style = L2_normalize(decoder_weights[:, top_K] @ style_scores[top_K])

Also saves the mean-shift direction v_shift = normalize(mean(z_tweet) - mean(z_formal))
as a trivial baseline (used for condition B2.5 in classify.py).

Run from project/ directory:
  python sae/sae_analysis.py [--top_k 32]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import SAEConfig
from sae.sae import SparseAutoencoder, load_sae

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


def compute_mean_activations(
    model: SparseAutoencoder,
    embeddings: torch.Tensor,
    batch_size: int = 512,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute mean SAE activations over a corpus of embeddings.
    Returns (hidden_dim,) mean activation vector.
    """
    model.eval()
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size].to(device)
            acts  = model.encode(batch)          # (B, hidden_dim)
            all_acts.append(acts.cpu())
    return torch.cat(all_acts, dim=0).mean(dim=0)   # (hidden_dim,)


def extract_style_direction(
    model: SparseAutoencoder,
    z_tweet: torch.Tensor,
    z_formal: torch.Tensor,
    top_k: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      v_style        (768,)   — weighted decoder direction for tweet style (aggregated)
      style_scores   (hidden_dim,) — raw per-feature style scores
      style_anchors  (K, 768) — individual top-K decoder columns (tweet-differential)
      anti_anchors   (K, 768) — individual bottom-K decoder columns (formal-differential)

    style_anchors / anti_anchors are the NLP analog of CLIP's mix_pairs:
    instead of a single fixed v_style, the PGD loss uses K paired anchors so
    the gradient is contrastive and multi-directional (mirrors the structure of
    perturb_bafa_txt_multi_ablation_lb_ls in the original CLIP code).
    """
    print("Computing mean activations on tweet corpus ...")
    tweet_mean_acts  = compute_mean_activations(model, z_tweet,  device=device)
    print("Computing mean activations on formal corpus ...")
    formal_mean_acts = compute_mean_activations(model, z_formal, device=device)

    style_scores = tweet_mean_acts - formal_mean_acts   # (hidden_dim,)

    # top-K features most differentially active on tweets vs formal
    top_k_indices = torch.argsort(style_scores, descending=True)[:top_k]
    top_k_scores  = style_scores[top_k_indices]         # (K,)

    # bottom-K features most differentially active on formal vs tweets (anti-style)
    bottom_k_indices = torch.argsort(style_scores, descending=False)[:top_k]

    print(f"  Top-{top_k} style features | max_score={top_k_scores[0]:.4f} | "
          f"min_score={top_k_scores[-1]:.4f}")

    # Decoder columns for top-K features: (input_dim, K) — pull to CPU for indexing
    decoder_weights = model.decoder.weight.data.cpu()    # (input_dim, hidden_dim)
    top_k_decoder   = decoder_weights[:, top_k_indices]  # (input_dim, K)
    bot_k_decoder   = decoder_weights[:, bottom_k_indices]  # (input_dim, K)

    # Weighted sum of decoder columns → style direction (aggregated, for diagnostics)
    v_style = top_k_decoder @ top_k_scores               # (input_dim,)
    v_style = F.normalize(v_style, dim=-1)

    # Individual L2-normalized decoder columns as paired anchor matrices
    style_anchors = F.normalize(top_k_decoder.T, dim=-1)  # (K, input_dim)
    anti_anchors  = F.normalize(bot_k_decoder.T, dim=-1)  # (K, input_dim)

    return v_style, style_scores, style_anchors, anti_anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k",     type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=1536)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load cached embeddings ---------------------------------------------
    tweet_path  = os.path.join(CACHE_DIR, "z_tweet_train.pt")
    formal_path = os.path.join(CACHE_DIR, "z_formal.pt")
    sae_ckpt    = os.path.join(CACHE_DIR, "sae_checkpoint.pt")

    if not os.path.exists(sae_ckpt):
        raise FileNotFoundError("SAE checkpoint not found. Run `python sae/sae.py` first.")

    z_tweet  = torch.load(tweet_path,  map_location="cpu")["embeddings"]
    z_formal = torch.load(formal_path, map_location="cpu")["embeddings"]
    print(f"z_tweet={tuple(z_tweet.shape)}  z_formal={tuple(z_formal.shape)}")

    # ---- Load SAE -----------------------------------------------------------
    cfg = SAEConfig(
        hidden_dim      = args.hidden_dim,
        top_k           = args.top_k,
        checkpoint_path = sae_ckpt,
        v_style_path    = os.path.join(CACHE_DIR, "v_style.pt"),
    )
    model = load_sae(cfg, device=device)
    print(f"SAE loaded: input={model.decoder.weight.shape[0]} "
          f"hidden={model.decoder.weight.shape[1]}")

    # ---- Extract v_style + anchor matrices ----------------------------------
    v_style, style_scores, style_anchors, anti_anchors = extract_style_direction(
        model, z_tweet, z_formal, top_k=args.top_k, device=device
    )
    print(f"\nv_style shape: {tuple(v_style.shape)}")
    print(f"style_anchors shape: {tuple(style_anchors.shape)}  "
          f"anti_anchors shape: {tuple(anti_anchors.shape)}")

    # ---- Compute mean-shift baseline ----------------------------------------
    v_shift = F.normalize(z_tweet.mean(0) - z_formal.mean(0), dim=-1)
    print(f"v_shift (mean-shift baseline) computed.")

    # ---- Sanity checks ------------------------------------------------------
    tweet_centroid  = F.normalize(z_tweet.mean(0),  dim=-1)
    formal_centroid = F.normalize(z_formal.mean(0), dim=-1)

    cos_style_tweet  = F.cosine_similarity(v_style.unsqueeze(0), tweet_centroid.unsqueeze(0)).item()
    cos_style_formal = F.cosine_similarity(v_style.unsqueeze(0), formal_centroid.unsqueeze(0)).item()
    print(f"\nSanity | cosine(v_style, tweet_centroid)  = {cos_style_tweet:.4f}")
    print(f"Sanity | cosine(v_style, formal_centroid) = {cos_style_formal:.4f}")
    if cos_style_tweet > cos_style_formal:
        print("  ✓ v_style aligns more with tweets than formal (expected)")
    else:
        print("  ⚠ v_style does NOT align more with tweets — check top_k or SAE training")

    # Print top-10 style feature statistics
    print(f"\nTop-10 style feature indices and scores:")
    top10 = torch.argsort(style_scores, descending=True)[:10]
    for rank, idx in enumerate(top10):
        print(f"  rank {rank+1:2d} | feature {idx.item():4d} | score {style_scores[idx].item():.4f}")

    # ---- Save ---------------------------------------------------------------
    torch.save(v_style,        os.path.join(CACHE_DIR, "v_style.pt"))
    torch.save(v_shift,        os.path.join(CACHE_DIR, "v_shift.pt"))
    torch.save(style_scores,   os.path.join(CACHE_DIR, "style_scores.pt"))
    torch.save(style_anchors,  os.path.join(CACHE_DIR, "style_anchors.pt"))
    torch.save(anti_anchors,   os.path.join(CACHE_DIR, "anti_style_anchors.pt"))

    print(f"\nSaved:")
    print(f"  v_style.pt             → {os.path.join(CACHE_DIR, 'v_style.pt')}")
    print(f"  v_shift.pt             → {os.path.join(CACHE_DIR, 'v_shift.pt')}")
    print(f"  style_scores.pt        → {os.path.join(CACHE_DIR, 'style_scores.pt')}")
    print(f"  style_anchors.pt       → {os.path.join(CACHE_DIR, 'style_anchors.pt')}  shape={tuple(style_anchors.shape)}")
    print(f"  anti_style_anchors.pt  → {os.path.join(CACHE_DIR, 'anti_style_anchors.pt')}  shape={tuple(anti_anchors.shape)}")
    print("SAE analysis complete.")


if __name__ == "__main__":
    main()
