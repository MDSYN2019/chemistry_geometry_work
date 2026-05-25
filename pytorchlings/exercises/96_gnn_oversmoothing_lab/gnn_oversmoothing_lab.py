"""Exercise 96: Oversmoothing in deep message passing.

Goals:
- repeatedly propagate over a graph and measure feature collapse
- track pairwise cosine similarity as depth increases
- try residual/skip variants to reduce collapse
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def row_normalize(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / deg


def mean_propagation_step(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return p @ x


def average_offdiag_cosine(x: torch.Tensor) -> float:
    z = F.normalize(x, dim=-1)
    sims = z @ z.T
    n = sims.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    return sims[mask].mean().item()


def run_oversmoothing_demo(num_layers: int = 20) -> list[float]:
    # 5-node line graph
    adj = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )

    p = row_normalize(adj)
    x = torch.eye(5)

    smoothness = []
    for _ in range(num_layers):
        x = mean_propagation_step(x, p)
        smoothness.append(average_offdiag_cosine(x))

    return smoothness


if __name__ == "__main__":
    smoothness_curve = run_oversmoothing_demo(num_layers=20)
    assert smoothness_curve[0] < smoothness_curve[-1], "expected similarity growth"

    print("first 5 cosine means:", [round(v, 4) for v in smoothness_curve[:5]])
    print("last 5 cosine means:", [round(v, 4) for v in smoothness_curve[-5:]])
    # TODO: add residual mixing: x <- alpha*x0 + (1-alpha)*(P@x) and compare curves.
    print("exercise 96 scaffold ready")
