import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
