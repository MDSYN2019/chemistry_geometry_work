"""Solution 83: residual MLP backbone."""

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ResidualMLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.out_norm(h)).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(83)
    x = torch.randn(10, 256)
    model = ResidualMLPRegressor(in_dim=256, hidden=96, depth=3)
    y = model(x)
    assert y.shape == (10,)
    print("solution 83 smoke check passed")
