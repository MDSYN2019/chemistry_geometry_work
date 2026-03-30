import torch
from torch import nn


def train_epoch_accum(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    accumulation_steps: int = 4,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running = 0.0

    optimizer.zero_grad()
    for step_idx, (x, y) in enumerate(loader):
        logits = model(x)
        loss = criterion(logits, y) / accumulation_steps
        loss.backward()

        if (step_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running += loss.item() * accumulation_steps

    if (step_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return running / max(1, step_idx + 1)
