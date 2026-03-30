"""Exercise 39: move tensors/model to the same device + dtype."""
import torch
from torch import nn


def prepare_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move x,y to model device and cast x to float32, y to int64."""
    # TODO: infer target device from model parameters
    device = torch.device("cpu")

    # TODO: move/cast tensors correctly
    x = x
    y = y
    return x, y
