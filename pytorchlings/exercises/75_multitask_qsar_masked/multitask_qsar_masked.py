"""Exercise 75: multitask QSAR with missing-label masking.

Goal:
- implement shared-trunk multitask prediction
- compute masked loss where some task labels are missing
"""

import torch
from torch import nn


class MultiTaskQSAR(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_tasks: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask shape: [batch, n_tasks], values in {0,1}
    sqerr = (pred - target) ** 2

    # TODO: support task-wise weighting so rare tasks contribute fairly
    denom = mask.sum().clamp_min(1.0)
    return (sqerr * mask).sum() / denom


if __name__ == "__main__":
    torch.manual_seed(75)
    bsz, in_dim, n_tasks = 16, 128, 5
    x = torch.randn(bsz, in_dim)
    y = torch.randn(bsz, n_tasks)
    mask = (torch.rand(bsz, n_tasks) > 0.3).float()

    model = MultiTaskQSAR(in_dim=in_dim, hidden=64, n_tasks=n_tasks)
    pred = model(x)
    loss = masked_mse(pred, y, mask)

    assert pred.shape == (bsz, n_tasks)
    assert torch.isfinite(loss)
    print("exercise 75 smoke check passed")
