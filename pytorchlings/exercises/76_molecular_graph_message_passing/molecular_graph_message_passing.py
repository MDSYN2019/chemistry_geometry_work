"""Exercise 76: molecular graph message passing without external graph libs.

Goal:
- understand neighborhood aggregation mechanics
- build a graph-level regressor with pooling
"""

import torch
from torch import nn


class SimpleMessagePassingLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh_mean = (adj @ x) / deg
        return torch.relu(self.self_lin(x) + self.neigh_lin(neigh_mean))


class MolecularGraphRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.mp1 = SimpleMessagePassingLayer(in_dim, hidden)
        self.mp2 = SimpleMessagePassingLayer(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.mp1(x, adj)
        h = self.mp2(h, adj)

        # TODO: replace mean pooling with attention/readout mechanism
        graph_repr = h.mean(dim=0, keepdim=True)
        return self.out(graph_repr).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(76)
    n, f = 24, 16
    x = torch.randn(n, f)
    a = torch.randint(0, 2, (n, n), dtype=torch.float32)
    adj = torch.triu(a, diagonal=1)
    adj = adj + adj.T
    adj.fill_diagonal_(1.0)

    model = MolecularGraphRegressor(in_dim=f, hidden=64)
    pred = model(x, adj)

    assert pred.numel() == 1
    print("exercise 76 smoke check passed")
