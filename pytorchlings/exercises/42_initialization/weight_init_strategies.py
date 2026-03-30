"""Exercise 42: apply module-wise parameter initialization."""
import torch
from torch import nn


class InitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def init_weights(model: InitNet) -> None:
    # TODO: apply kaiming_uniform_ to Linear.weight and zeros_ to Linear.bias
    pass
