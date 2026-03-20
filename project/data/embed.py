"""
Phase 1: Encode all corpora and cache embeddings to disk.

Outputs (saved to project/cache/):
  z_tweet_train.pt  — dict {"embeddings": (N,768), "labels": (N,)}
  z_tweet_val.pt
  z_tweet_test.pt
  z_formal.pt       — dict {"embeddings": (N,768)}  (no labels)

Run from project/ directory:
  python data/embed.py [--batch_size 64] [--max_length 128]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from encoder import FinBERTEncoder
from dataset import load_tsad, load_formal_sentences


_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


def encode_texts(encoder: FinBERTEncoder, texts: list[str], batch_size: int, max_length: int) -> torch.Tensor:
    """Encode a list of texts in batches. Returns (N, H) L2-normalized tensor."""
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  encoding batches"):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            vecs = encoder.encode_text(batch, max_length=max_length)  # (B, H)
        all_vecs.append(vecs.cpu())
    return torch.cat(all_vecs, dim=0)  # (N, H)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "ProsusAI/finbert"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---- Load encoder --------------------------------------------------------
    try:
        encoder = FinBERTEncoder(model_name=args.model_name, device=device)
    except Exception:
        print(f"Falling back to distilbert-base-uncased")
        encoder = FinBERTEncoder(model_name="distilbert-base-uncased", device=device)

    # ---- Load tweet data -----------------------------------------------------
    tr_t, tr_l, va_t, va_l, te_t, te_l = load_tsad()

    for split_name, texts, labels in [
        ("train", tr_t, tr_l),
        ("val",   va_t, va_l),
        ("test",  te_t, te_l),
    ]:
        out_path = os.path.join(CACHE_DIR, f"z_tweet_{split_name}.pt")
        print(f"\nEncoding tweet {split_name} ({len(texts)} samples) ...")
        embs = encode_texts(encoder, texts, args.batch_size, args.max_length)
        torch.save(
            {"embeddings": embs, "labels": torch.tensor(labels, dtype=torch.long)},
            out_path,
        )
        print(f"  Saved → {out_path}  shape={tuple(embs.shape)}")

    # ---- Load formal sentences -----------------------------------------------
    formal_texts = load_formal_sentences()
    out_path = os.path.join(CACHE_DIR, "z_formal.pt")
    print(f"\nEncoding formal sentences ({len(formal_texts)} samples) ...")
    embs = encode_texts(encoder, formal_texts, args.batch_size, args.max_length)
    torch.save({"embeddings": embs}, out_path)
    print(f"  Saved → {out_path}  shape={tuple(embs.shape)}")

    # ---- Sanity check --------------------------------------------------------
    print("\n--- Sanity check ---")
    z_tw = torch.load(os.path.join(CACHE_DIR, "z_tweet_train.pt"))["embeddings"]
    z_fo = torch.load(out_path)["embeddings"]

    # Sample 500 random pairs
    n = min(500, len(z_tw), len(z_fo))
    idx_tw = torch.randperm(len(z_tw))[:n]
    idx_fo = torch.randperm(len(z_fo))[:n]
    sim_tw_tw = F.cosine_similarity(z_tw[idx_tw], z_tw[idx_fo[:n]], dim=-1).mean().item()
    sim_tw_fo = F.cosine_similarity(z_tw[idx_tw], z_fo[idx_fo], dim=-1).mean().item()
    print(f"  mean cosine(tweet, tweet) = {sim_tw_tw:.4f}")
    print(f"  mean cosine(tweet, formal) = {sim_tw_fo:.4f}")
    if sim_tw_fo < sim_tw_tw:
        print("  ✓ Domain shift confirmed (tweet-formal gap exists)")
    else:
        print("  ⚠ No domain shift detected — style removal may not help")

    print("\nEmbedding extraction complete.")


if __name__ == "__main__":
    main()
