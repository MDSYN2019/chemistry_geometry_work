"""Exercise 40: evaluation mode + no_grad inference."""
import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run inference with dropout/batchnorm frozen and without grad tracking."""
    # TODO: switch to eval mode

    # TODO: run forward pass under torch.no_grad()
    return model(x)
