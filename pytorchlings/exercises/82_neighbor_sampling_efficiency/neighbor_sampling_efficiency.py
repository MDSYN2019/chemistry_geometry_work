"""Exercise 82: efficiency with full-batch vs neighbor sampling.

Goal:
- compare memory/time tradeoffs in large-graph training
- run identical model under different batching regimes
- quantify the accuracy-vs-throughput frontier
"""

from __future__ import annotations

import time

import torch
from torch_geometric.data import Data


class TinyEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_denseish_graph(n: int = 3000, d: int = 32) -> Data:
    torch.manual_seed(82)
    x = torch.randn(n, d)
    src = torch.randint(0, n, (n * 8,))
    dst = torch.randint(0, n, (n * 8,))
    edge_index = torch.stack([src, dst], dim=0)
    y = torch.randint(0, 3, (n,))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[: int(0.7 * n)] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)


def run_full_batch_stub(data: Data) -> float:
    model = TinyEncoder(data.x.size(1), 64, 3)
    start = time.perf_counter()
    _ = model(data.x)
    return time.perf_counter() - start


def run_neighbor_sampling_stub(data: Data) -> float:
    """TODO: replace with NeighborLoader-based mini-batch training runtime."""
    return run_full_batch_stub(data)


if __name__ == "__main__":
    data = make_denseish_graph()
    t_full = run_full_batch_stub(data)
    t_sample = run_neighbor_sampling_stub(data)

    assert t_full >= 0.0 and t_sample >= 0.0
    print(f"exercise 82 scaffold ready | full={t_full:.6f}s sampled={t_sample:.6f}s")
