"""
Phase 2: Sparse Autoencoder (SAE) for tweet-style direction discovery.

Trained on the mixed corpus of tweet + formal embeddings.
Finds an overcomplete dictionary (768 → 1536 → 768) with L1 sparsity.

Run from project/ directory:
  python sae/sae.py [--epochs 50] [--hidden_dim 1536] [--lambda_l1 1e-3]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import SAEConfig

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_DIR = os.environ.get("CACHE_DIR", _DEFAULT_CACHE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SparseAutoencoder(nn.Module):
    """
    Overcomplete sparse autoencoder for FinBERT CLS embeddings.

    encoder: Linear(768, hidden_dim) + ReLU  →  activations f(z)
    decoder: Linear(hidden_dim, 768, bias=False)  →  reconstruction z_hat

    Loss: MSE(z, z_hat) + lambda_l1 * ||f(z)||_1
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 1536):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # No bias on decoder so that features have a clean geometric interpretation
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, z: torch.Tensor):
        """
        z       : (B, input_dim)  L2-normalized input
        returns : activations (B, hidden_dim), reconstruction (B, input_dim)
        """
        activations = self.encoder(z)          # (B, hidden_dim)
        reconstruction = self.decoder(activations)  # (B, input_dim)
        return activations, reconstruction

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Return only the sparse activations (B, hidden_dim)."""
        return self.encoder(z)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_sae(cfg: SAEConfig, device: str = "cpu") -> SparseAutoencoder:
    """Load cached embeddings, train SAE, save checkpoint."""

    # ---- Load cached embeddings ---------------------------------------------
    tweet_path  = os.path.join(CACHE_DIR, "z_tweet_train.pt")
    formal_path = os.path.join(CACHE_DIR, "z_formal.pt")

    if not os.path.exists(tweet_path) or not os.path.exists(formal_path):
        raise FileNotFoundError(
            "Cached embeddings not found. Run `python data/embed.py` first."
        )

    z_tweet  = torch.load(tweet_path,  map_location="cpu")["embeddings"]   # (N_t, 768)
    z_formal = torch.load(formal_path, map_location="cpu")["embeddings"]   # (N_f, 768)

    print(f"Loaded: z_tweet={tuple(z_tweet.shape)}  z_formal={tuple(z_formal.shape)}")

    # Mix the two corpora
    z_all = torch.cat([z_tweet, z_formal], dim=0)
    print(f"Mixed corpus: {len(z_all)} samples")

    # ---- Shuffle and create DataLoader --------------------------------------
    perm = torch.randperm(len(z_all))
    z_all = z_all[perm]

    dataset = TensorDataset(z_all)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # ---- Model, optimizer ---------------------------------------------------
    input_dim = z_all.shape[1]
    model = SparseAutoencoder(input_dim=input_dim, hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    print(f"\nTraining SAE: input={input_dim} → hidden={cfg.hidden_dim} → {input_dim}")
    print(f"  epochs={cfg.epochs}  batch={cfg.batch_size}  lambda_l1={cfg.lambda_l1}")

    best_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_mse = 0.0
        total_l1  = 0.0
        n_batches = 0

        for (z_batch,) in loader:
            z_batch = z_batch.to(device)
            activations, reconstruction = model(z_batch)

            mse_loss = F.mse_loss(reconstruction, z_batch)
            l1_loss  = activations.abs().mean()
            loss     = mse_loss + cfg.lambda_l1 * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse += mse_loss.item()
            total_l1  += l1_loss.item()
            n_batches += 1

        avg_mse = total_mse / n_batches
        avg_l1  = total_l1  / n_batches

        if epoch % 5 == 0 or epoch == 1:
            # Compute sparsity: % of features active per sample (activation > 0)
            model.eval()
            with torch.no_grad():
                sample = z_all[:1000].to(device)
                acts   = model.encode(sample)
                sparsity_pct = (acts > 0).float().mean().item() * 100
            print(f"  epoch {epoch:3d}/{cfg.epochs} | MSE={avg_mse:.4f} | "
                  f"L1={avg_l1:.4f} | active={sparsity_pct:.1f}%")

        if avg_mse < best_loss:
            best_loss = avg_mse

    # ---- Save checkpoint ----------------------------------------------------
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), cfg.checkpoint_path)
    print(f"\nSAE saved → {cfg.checkpoint_path}")
    print(f"Final reconstruction MSE: {best_loss:.5f}")

    return model


def load_sae(cfg: SAEConfig, device: str = "cpu") -> SparseAutoencoder:
    """Load a trained SAE from checkpoint."""
    # Infer input_dim from the first cached embedding
    sample = torch.load(os.path.join(CACHE_DIR, "z_tweet_train.pt"), map_location="cpu")
    input_dim = sample["embeddings"].shape[1]

    model = SparseAutoencoder(input_dim=input_dim, hidden_dim=cfg.hidden_dim)
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim",  type=int,   default=1536)
    parser.add_argument("--lambda_l1",  type=float,  default=1e-3)
    parser.add_argument("--epochs",     type=int,    default=50)
    parser.add_argument("--batch_size", type=int,    default=256)
    parser.add_argument("--lr",         type=float,  default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = SAEConfig(
        hidden_dim       = args.hidden_dim,
        lambda_l1        = args.lambda_l1,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        checkpoint_path  = os.path.join(CACHE_DIR, "sae_checkpoint.pt"),
        v_style_path     = os.path.join(CACHE_DIR, "v_style.pt"),
    )

    train_sae(cfg, device=device)
    print("Done.")


if __name__ == "__main__":
    main()
