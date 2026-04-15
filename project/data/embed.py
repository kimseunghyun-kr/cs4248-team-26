"""
Phase 1: Encode all corpora and cache embeddings to disk.

Outputs (saved to cache directory):
  z_tweet_train.pt  — dict {"embeddings": (N,H), "labels": (N,),
                             "input_ids": (N,L), "attention_mask": (N,L),
                             "texts": list[str], "entities": list[str|None],
                             "cleaned_tokens": list[list[str]|None],
                             "selected_texts": list[str|None]}
  z_tweet_val.pt
  z_tweet_test.pt

Run from project/ directory:
  python data/embed.py [--batch_size 64] [--max_length 128]
  EMBED_BATCH_SIZE=4 python data/embed.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from tqdm import tqdm

from encoder import TransformerEncoder
from dataset import load_records


_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


def encode_texts_with_tokens(
    encoder: TransformerEncoder,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode texts and return (embeddings, input_ids, attention_mask).
    padding='max_length' ensures uniform shape across batches for torch.cat.
    Returns:
        embeddings    : (N, H)  L2-normalized CLS vectors
        input_ids     : (N, L)  token ids
        attention_mask: (N, L)  1/0 mask
    """
    all_vecs, all_ids, all_masks = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="  encoding batches"):
        batch = texts[i : i + batch_size]
        enc = encoder.tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(encoder.device)
        mask = enc["attention_mask"].to(encoder.device)
        with torch.no_grad():
            embeds = encoder.encode_ids(ids, mask)  # (B, H)
        all_vecs.append(embeds.cpu())
        all_ids.append(ids.cpu())
        all_masks.append(mask.cpu())

    return torch.cat(all_vecs), torch.cat(all_ids), torch.cat(all_masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "bert-base-uncased"))
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("EMBED_BATCH_SIZE", "64")))
    parser.add_argument("--max_length", type=int, default=int(os.environ.get("EMBED_MAX_LENGTH", "128")))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---- Load encoder (with optional custom tokenizer) -----------------------
    tokenizer = None
    tokenizer_name = os.environ.get("TOKENIZER_NAME")
    if tokenizer_name:
        from transformers import AutoTokenizer
        print(f"Loading custom tokenizer from '{tokenizer_name}' ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    try:
        encoder = TransformerEncoder(model_name=args.model_name, device=device, tokenizer=tokenizer)
    except Exception as e:
        print(f"ERROR: failed to load requested model '{args.model_name}': {e}")
        raise

    # ---- Load text data ------------------------------------------------------
    train_records, val_records, test_records = load_records()

    for split_name, records in [
        ("train", train_records),
        ("val",   val_records),
        ("test",  test_records),
    ]:
        texts = [r["text"] for r in records]
        labels = [r["label"] for r in records]
        out_path = os.path.join(CACHE_DIR, f"z_tweet_{split_name}.pt")
        print(f"\nEncoding {split_name} ({len(texts)} samples) ...")
        embs, ids, masks = encode_texts_with_tokens(
            encoder, texts, args.batch_size, args.max_length
        )
        torch.save(
            {
                "embeddings":     embs,
                "labels":         torch.tensor(labels, dtype=torch.long),
                "input_ids":      ids,
                "attention_mask": masks,
                "texts":          texts,
                "entities":       [r.get("entity") for r in records],
                "cleaned_tokens": [r.get("cleaned_tokens") for r in records],
                "selected_texts": [r.get("selected_text") for r in records],
                "time_of_tweet":  [r.get("time_of_tweet") for r in records],
                "age_of_user":    [r.get("age_of_user") for r in records],
                "country":        [r.get("country") for r in records],
            },
            out_path,
        )
        print(f"  Saved → {out_path}  shape={tuple(embs.shape)}")

    print("\nEmbedding extraction complete.")


if __name__ == "__main__":
    main()
