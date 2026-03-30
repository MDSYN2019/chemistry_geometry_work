import torch
from torch import nn


def amp_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    model.train()
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cpu")

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(x)
        loss = criterion(logits, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item())
