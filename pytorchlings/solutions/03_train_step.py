import torch
from torch import nn


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, optim: torch.optim.Optimizer) -> float:
    model.train()
    optim.zero_grad()
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()
    optim.step()
    return float(loss.item())
