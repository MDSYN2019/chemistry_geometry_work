"""Exercise 02: build a 2-layer MLP."""
import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 2):
        super().__init__()
        # TODO: replace Identity with Linear/ReLU/Linear stack
        self.net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
