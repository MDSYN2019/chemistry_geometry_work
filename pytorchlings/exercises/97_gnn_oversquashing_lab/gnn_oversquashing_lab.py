"""Exercise 97: Oversquashing and graph bottlenecks.

Goals:
- build a bottleneck graph where many sources route through one bridge node
- compare information retention at the target as source count increases
- test mitigation ideas (rewiring, virtual edges, global attention)
"""

from __future__ import annotations

import torch


def bottleneck_aggregate(source_features: torch.Tensor) -> torch.Tensor:
    """Compress many source vectors into one bridge message.

    This intentionally uses mean aggregation to illustrate an information
    bottleneck before forwarding to the target node.
    """
    return source_features.mean(dim=0)


def retention_proxy(source_features: torch.Tensor) -> float:
    """Heuristic: reconstruction error from only the bottleneck mean.

    Lower is better. Error typically rises as source count/diversity rises.
    """
    bridge = bottleneck_aggregate(source_features)
    recon = bridge.unsqueeze(0).expand_as(source_features)
    mse = torch.mean((source_features - recon) ** 2).item()
    return mse


def run_bottleneck_scaling(dim: int = 16) -> list[tuple[int, float]]:
    torch.manual_seed(0)
    out = []
    for n_sources in [2, 4, 8, 16, 32, 64]:
        x = torch.randn(n_sources, dim)
        out.append((n_sources, retention_proxy(x)))
    return out


if __name__ == "__main__":
    scaling = run_bottleneck_scaling()
    print("source_count -> bottleneck mse")
    for n, err in scaling:
        print(f"{n:>2} -> {err:.4f}")

    # Monotonicity is not guaranteed for one sample, but broad trend should rise.
    assert scaling[0][1] < scaling[-1][1]
    # TODO: add rewiring experiment: split sources across multiple bridge nodes and
    # compare bottleneck MSE to single-bridge baseline.
    print("exercise 97 scaffold ready")
