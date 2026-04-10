"""Exercise 71: compare graph-only vs geometry-aware protein-ligand models."""

import torch
from torch import nn


class GraphOnlyScorer(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        pooled = node_features.mean(dim=0, keepdim=True)
        return self.mlp(pooled).squeeze(-1)


class GeometryAwareScorer(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.feature_proj = nn.Linear(in_dim, hidden)
        self.dist_proj = nn.Linear(1, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, node_features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        pairwise_dist = torch.cdist(coords, coords)
        geom_signal = pairwise_dist.mean(dim=1, keepdim=True)

        # TODO: replace mean distance with richer geometric basis features
        h = self.feature_proj(node_features) + self.dist_proj(geom_signal)
        pooled = h.mean(dim=0, keepdim=True)
        return self.head(torch.relu(pooled)).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    n, f = 40, 20
    x = torch.randn(n, f)
    xyz = torch.randn(n, 3)

    g = GraphOnlyScorer(in_dim=f, hidden=64)
    geom = GeometryAwareScorer(in_dim=f, hidden=64)

    s1 = g(x)
    s2 = geom(x, xyz)
    assert s1.numel() == 1 and s2.numel() == 1
    print("exercise 71 smoke check passed")
