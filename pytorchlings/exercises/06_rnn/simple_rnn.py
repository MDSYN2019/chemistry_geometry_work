"""Exercise 06: simple GRU classifier."""
import torch
from torch import nn


class SequenceClassifier(nn.Module):
    def __init__(self, input_size: int = 6, hidden: int = 12, n_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: use final hidden state from GRU instead of mean
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.fc(pooled)
