import torch
from torch import nn


def export_torchscript(model: nn.Module, example: torch.Tensor) -> torch.jit.ScriptModule:
    try:
        return torch.jit.script(model)
    except Exception:
        return torch.jit.trace(model, example)
