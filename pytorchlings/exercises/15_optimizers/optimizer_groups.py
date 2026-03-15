"""Exercise 15: optimizer parameter groups."""
import torch
from torch import nn


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_optimizer(model: SmallNet) -> torch.optim.Optimizer:
    # TODO: set lr=1e-4 for backbone and lr=1e-3 for head
    return torch.optim.Adam(model.parameters(), lr=1e-3)
