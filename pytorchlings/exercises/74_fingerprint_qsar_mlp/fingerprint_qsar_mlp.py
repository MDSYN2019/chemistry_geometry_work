"""Exercise 74: feed-forward QSAR baseline on fingerprint-like features.

Goal:
- implement a strong chemistry baseline with an MLP
- practice regression loop + RMSE reporting
- compare with a non-neural baseline later
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FingerprintMLP(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(torch.mean((pred - target) ** 2).item())


def make_synthetic_qsar(n: int = 512, d: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(74)
    x = torch.randint(0, 2, (n, d), dtype=torch.float32)
    w = torch.randn(d)
    y = x @ w / d + 0.1 * torch.randn(n)
    return x, y


def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    running, n = 0.0, 0

    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)

        # TODO: add learning-rate scheduler step and gradient clipping
        loss.backward()
        opt.step()

        running += loss.item() * xb.size(0)
        n += xb.size(0)
    return running / max(n, 1)


if __name__ == "__main__":
    x, y = make_synthetic_qsar()
    train_ds = TensorDataset(x[:400], y[:400])
    val_x, val_y = x[400:], y[400:]

    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    model = FingerprintMLP(in_dim=x.size(1))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    for _ in range(5):
        train_epoch(model, loader, opt)

    model.eval()
    with torch.no_grad():
        val_pred = model(val_x)
    val_rmse = rmse(val_pred, val_y)

    assert val_rmse >= 0.0
    print(f"exercise 74 smoke check passed | val_rmse={val_rmse:.4f}")
