"""Exercise 67: implement a paper baseline without high-level wrappers.

Focus:
- explicit model definition
- explicit train/eval loops
- gradient/debug instrumentation
- basic ablation hook points
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BaselineMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainStats:
    train_loss: float
    val_loss: float
    val_acc: float


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion: nn.Module) -> float:
    model.train()
    running, n = 0.0, 0

    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        # TODO: add optional gradient norm logging before optimizer.step()
        loss.backward()
        optimizer.step()

        running += loss.item() * x.size(0)
        n += x.size(0)

    return running / max(n, 1)


def run_baseline(
    epochs: int = 5,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
) -> TrainStats:
    torch.manual_seed(7)
    x = torch.randn(512, 16)
    y = torch.randint(0, 4, (512,))

    train_ds = TensorDataset(x[:400], y[:400])
    val_ds = TensorDataset(x[400:], y[400:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = BaselineMLP(in_dim=16, hidden=64, out_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    last_train = 0.0
    for _ in range(epochs):
        last_train = train_one_epoch(model, train_loader, optimizer, criterion)

    val_loss, val_acc = evaluate(model, val_loader, criterion)
    return TrainStats(train_loss=last_train, val_loss=val_loss, val_acc=val_acc)


if __name__ == "__main__":
    stats = run_baseline()
    assert 0.0 <= stats.val_acc <= 1.0
    print(stats)
