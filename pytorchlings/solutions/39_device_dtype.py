import torch
from torch import nn


def prepare_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.int64)
    return x, y
