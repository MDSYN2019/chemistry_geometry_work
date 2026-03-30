"""Exercise 44: mixed precision training step with GradScaler."""
import torch
from torch import nn


def amp_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """Run one training step using autocast + GradScaler."""
    model.train()
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cpu")

    # TODO: wrap forward/loss with torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    logits = model(x)
    loss = criterion(logits, y)

    # TODO: scale(loss).backward(), scaler.step(optimizer), scaler.update()
    loss.backward()
    optimizer.step()
    return float(loss.item())
