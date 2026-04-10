"""Exercise 4: BioAI multi-task model (toy setup).

Goal:
Train a model that predicts three targets from a sequence embedding:
- affinity_score (regression)
- developability_flag (binary)
- toxicity_proxy (regression)

Task:
- Implement weighted multi-task loss.
- Add metric tracking per head.
- Propose how this toy setup maps to a realistic wet-lab loop.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MultiTaskNet(nn.Module):
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.affinity_head = nn.Linear(128, 1)
        self.dev_head = nn.Linear(128, 1)
        self.tox_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return {
            "affinity": self.affinity_head(h),
            "developability": self.dev_head(h),
            "toxicity": self.tox_head(h),
        }


def make_loader(n: int = 3000, input_dim: int = 128) -> DataLoader:
    x = torch.randn(n, input_dim)
    affinity = x[:, :8].sum(dim=1, keepdim=True) + 0.1 * torch.randn(n, 1)
    developability = (x[:, 8:16].sum(dim=1, keepdim=True) > 0).float()
    toxicity = x[:, 16:24].mean(dim=1, keepdim=True) + 0.1 * torch.randn(n, 1)
    ds = TensorDataset(x, affinity, developability, toxicity)
    return DataLoader(ds, batch_size=64, shuffle=True)


def main() -> None:
    model = MultiTaskNet()
    loader = make_loader()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    for epoch in range(3):
        running = 0.0
        for x, affinity, developability, toxicity in loader:
            out = model(x)

            # Candidate can tune weights and justify choices.
            loss = (
                1.0 * mse(out["affinity"], affinity)
                + 0.7 * bce(out["developability"], developability)
                + 0.8 * mse(out["toxicity"], toxicity)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())

        print(f"epoch={epoch} multitask_loss={running/len(loader):.4f}")


if __name__ == "__main__":
    main()
