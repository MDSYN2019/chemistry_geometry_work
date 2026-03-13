import torch
from torch import nn


class SequenceClassifier(nn.Module):
    def __init__(self, input_size: int = 6, hidden: int = 12, n_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        last = h_n[-1]
        return self.fc(last)
