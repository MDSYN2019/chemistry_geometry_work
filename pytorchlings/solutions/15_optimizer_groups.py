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
    return torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ])
