"""Exercise 3: Profile and optimize a slow training loop.

Task:
1) Run this script and capture baseline runtime.
2) Use torch.profiler (or cProfile) to identify bottlenecks.
3) Implement at least 2 optimizations and report impact.

Hints:
- increase batch size carefully
- avoid repeated tensor allocations in loop
- use num_workers/pin_memory when using CUDA
"""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_data() -> DataLoader:
    x = torch.randn(8000, 128)
    y = torch.randint(0, 2, (8000, 1)).float()
    ds = TensorDataset(x, y)
    # Intentionally conservative defaults for candidate to tune.
    return DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )


def main() -> None:
    loader = build_data()
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    for epoch in range(3):
        total = 0.0
        for x, y in loader:
            # Intentionally suboptimal pattern (extra allocation) for optimization exercise.
            x = x + torch.zeros_like(x)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch} loss={total/len(loader):.4f}")

    elapsed = time.perf_counter() - t0
    print(f"baseline_elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
