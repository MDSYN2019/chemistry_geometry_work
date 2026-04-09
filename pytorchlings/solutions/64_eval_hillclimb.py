from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass
class TrialResult:
    hidden_dim: int
    lr: float
    val_mae: float


def make_dataset(n: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(12)
    x = torch.randn(n, 6, generator=g)
    y = (0.6 * x[:, 0] - 0.3 * x[:, 1] ** 2 + 0.4 * x[:, 2] * x[:, 3]).unsqueeze(1)
    y = y + 0.05 * torch.randn_like(y, generator=g)
    return x, y


def train_one_trial(hidden_dim: int, lr: float, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor) -> TrialResult:
    model = nn.Sequential(nn.Linear(6, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(80):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        val_mae = torch.mean(torch.abs(model(x_val) - y_val)).item()
    return TrialResult(hidden_dim=hidden_dim, lr=lr, val_mae=float(val_mae))


def hillclimb_search() -> tuple[TrialResult, list[TrialResult]]:
    x, y = make_dataset()
    x_train, y_train = x[:400], y[:400]
    x_val, y_val = x[400:], y[400:]

    current = TrialResult(hidden_dim=16, lr=2e-3, val_mae=1e9)
    history: list[TrialResult] = []

    for _ in range(4):
        best_round: TrialResult | None = None
        dim_candidates = [current.hidden_dim, current.hidden_dim * 2]
        lr_candidates = [current.lr, current.lr * 0.5, current.lr * 2.0]

        for hd in dim_candidates:
            for lr in lr_candidates:
                candidate = train_one_trial(hd, lr, x_train, y_train, x_val, y_val)
                if best_round is None or candidate.val_mae < best_round.val_mae:
                    best_round = candidate

        assert best_round is not None
        current = best_round
        history.append(current)

    return current, history
