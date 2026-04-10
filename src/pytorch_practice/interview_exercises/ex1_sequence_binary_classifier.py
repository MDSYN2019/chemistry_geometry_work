"""Exercise 1: Sequence binary classification in PyTorch.

Task:
1) Implement both `MLPClassifier` and `TinyConvClassifier`.
2) Train both models on the synthetic dataset.
3) Compare validation AUC and discuss overfitting behavior.
4) Add one improvement (e.g., dropout, weight decay, scheduler).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class SyntheticSequenceDataset(Dataset):
    def __init__(self, n_samples: int = 4000, seq_len: int = 64, vocab_size: int = 24):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.samples = torch.randint(0, vocab_size, (n_samples, seq_len))

        # Toy label rule: positive if high count of tokens in a motif set.
        motif_tokens = {2, 7, 13, 19}
        counts = torch.zeros(n_samples)
        for t in motif_tokens:
            counts += (self.samples == t).sum(dim=1)
        self.labels = (counts > 12).float().unsqueeze(1)

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """TODO: candidate implements."""

    def __init__(self, vocab_size: int, seq_len: int, emb_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(x))


class TinyConvClassifier(nn.Module):
    """TODO: candidate compares against MLP baseline."""

    def __init__(self, vocab_size: int, emb_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Conv1d(emb_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x).transpose(1, 2)
        return self.net(z)


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 5


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item())
    return total / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total += float(loss_fn(logits, y).item())
    return total / len(loader)


def main() -> None:
    seed_everything(7)
    ds = SyntheticSequenceDataset()
    train_size = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = MLPClassifier(vocab_size=24, seq_len=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, 6):
        tr = train_one_epoch(model, train_loader, opt, loss_fn)
        va = evaluate(model, val_loader, loss_fn)
        print(f"epoch={epoch} train_loss={tr:.4f} val_loss={va:.4f}")


if __name__ == "__main__":
    main()
