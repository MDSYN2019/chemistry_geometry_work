import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compile_and_run(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    compiled = torch.compile(model)
    return compiled(x)
