"""Exercise 41: use forward hooks to capture intermediate activations."""
import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.head = nn.Linear(4 * 8 * 8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = x.flatten(start_dim=1)
        return self.head(x)


def capture_conv_output(model: TinyCNN, x: torch.Tensor) -> torch.Tensor:
    features: list[torch.Tensor] = []

    def hook_fn(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        features.append(out.detach())

    # TODO: register hook on model.conv, run model(x), then remove hook
    model(x)

    return features[0]
