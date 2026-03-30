"""Exercise 46: TorchScript scripting/tracing basics."""
import torch
from torch import nn


def export_torchscript(model: nn.Module, example: torch.Tensor) -> torch.jit.ScriptModule:
    """Return a TorchScript module generated from model."""
    # TODO: script or trace the model with example input
    return model  # type: ignore[return-value]
