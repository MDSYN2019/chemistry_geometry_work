"""Exercise 43: gradient accumulation across micro-batches."""
import torch
from torch import nn


def train_epoch_accum(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    accumulation_steps: int = 4,
) -> float:
    """Train for one epoch and step optimizer every accumulation_steps mini-batches."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    running = 0.0

    optimizer.zero_grad()
    for step_idx, (x, y) in enumerate(loader):
        logits = model(x)
        # TODO: divide by accumulation_steps before backward for equivalent gradients
        loss = criterion(logits, y)
        loss.backward()

        # TODO: step/zero grad every accumulation_steps

        running += loss.item()

    return running / max(1, step_idx + 1)
