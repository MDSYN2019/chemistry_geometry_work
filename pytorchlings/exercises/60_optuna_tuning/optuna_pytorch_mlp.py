"""Exercise 60: tune a small PyTorch MLP with Optuna.

Install dependency first:
    pip install optuna
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_toy_data(n: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary classification toy data (two noisy circles)."""
    g = torch.Generator().manual_seed(7)
    r = torch.rand(n, generator=g)
    theta = torch.rand(n, generator=g) * (2.0 * math.pi)
    x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    y = (r > 0.55).long()
    x = x + 0.05 * torch.randn_like(x, generator=g)
    return x, y


def build_model(hidden_dim: int, dropout: float) -> nn.Module:
    """Return a tiny MLP for 2D -> 2-class classification."""
    return nn.Sequential(
        nn.Linear(2, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 2),
    )


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Run one epoch and return mean loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * xb.size(0)
        total += xb.size(0)
    return running_loss / max(total, 1)


def evaluate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Return classification accuracy in [0, 1]."""
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean().item())


def objective(trial, train_loader: DataLoader, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    """Optuna objective: maximize validation accuracy."""
    # TODO: sample hidden_dim in [8, 128] (step=8) with suggest_int.
    hidden_dim = 32
    # TODO: sample dropout in [0.0, 0.5] with suggest_float.
    dropout = 0.1
    # TODO: sample learning rate log-uniform in [1e-4, 1e-1].
    lr = 1e-3

    model = build_model(hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TODO: sample number of epochs in [5, 30] and train for that many epochs.
    for _ in range(5):
        train_epoch(model, train_loader, optimizer)

    # TODO: report intermediate values for pruning and return final val accuracy.
    return evaluate_accuracy(model, x_val, y_val)


def run_study(n_trials: int = 20):
    """Create and run an Optuna study."""
    import optuna

    x, y = make_toy_data(n=1024)
    split = 768
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

    # TODO: use TPESampler with seed=123.
    sampler = None
    # TODO: use MedianPruner(n_startup_trials=5, n_warmup_steps=2).
    pruner = None

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="pytorch_mlp_tuning",
    )
    study.optimize(lambda trial: objective(trial, train_loader, x_val, y_val), n_trials=n_trials)
    return study


if __name__ == "__main__":
    study = run_study(n_trials=10)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
