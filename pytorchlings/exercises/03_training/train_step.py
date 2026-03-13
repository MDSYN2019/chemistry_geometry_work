"""Exercise 03: implement one training step."""
import torch
from torch import nn


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, optim: torch.optim.Optimizer) -> float:
    # TODO: add zero_grad -> forward -> loss -> backward -> step
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    return float(loss.item())
