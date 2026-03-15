"""Exercise 03: implement one training step."""
import torch
from torch import nn

optimizer = torch.optim.SGD

def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, optim: torch.optim.Optimizer) -> float:
    # TODO: add zero_grad -> forward -> loss -> backward -> step
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    # optimizer zero grad
    optim.zero_grad()
    # loss backwards
    loss.backward()
    optimizer.step()
    return float(loss.item())
