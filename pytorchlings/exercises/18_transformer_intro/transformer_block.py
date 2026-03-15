"""Exercise 18: minimal transformer encoder block."""
import torch
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: add residual connections around attention and feed-forward blocks
        h, _ = self.attn(x, x, x)
        return self.ff(h)
