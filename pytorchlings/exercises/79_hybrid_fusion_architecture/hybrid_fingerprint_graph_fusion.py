"""Exercise 79: hybrid fingerprint + graph architecture.

Builds on:
- Exercise 74 (fingerprint MLP baseline)
- Exercise 76 (message passing graph encoder)

Goal:
- encode tabular fingerprint and molecular graph views separately
- fuse both representations for final prediction
"""

import torch
from torch import nn


class FingerprintEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x_fp: torch.Tensor) -> torch.Tensor:
        return self.net(x_fp)


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden: int = 128):
        super().__init__()
        self.self_lin = nn.Linear(node_dim, hidden)
        self.neigh_lin = nn.Linear(node_dim, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x_nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x_nodes shape: [batch, n_nodes, node_dim]
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh_mean = (adj @ x_nodes) / deg
        h = torch.relu(self.self_lin(x_nodes) + self.neigh_lin(neigh_mean))

        # TODO: replace mean pool with attention or set-transformer readout
        graph_repr = h.mean(dim=1)
        return torch.relu(self.out(graph_repr))


class HybridFusionRegressor(nn.Module):
    def __init__(self, fp_dim: int, node_dim: int, hidden: int = 128):
        super().__init__()
        self.fp_encoder = FingerprintEncoder(fp_dim, hidden)
        self.graph_encoder = GraphEncoder(node_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_fp: torch.Tensor, x_nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        fp_repr = self.fp_encoder(x_fp)
        graph_repr = self.graph_encoder(x_nodes, adj)
        fused = torch.cat([fp_repr, graph_repr], dim=-1)
        return self.head(fused).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(79)
    batch, fp_dim, n_nodes, node_dim = 8, 256, 18, 24
    x_fp = torch.randn(batch, fp_dim)
    x_nodes = torch.randn(batch, n_nodes, node_dim)

    raw = torch.randint(0, 2, (batch, n_nodes, n_nodes), dtype=torch.float32)
    adj = torch.triu(raw, diagonal=1)
    adj = adj + adj.transpose(-1, -2)
    eye = torch.eye(n_nodes).unsqueeze(0)
    adj = (adj + eye).clamp(max=1.0)

    model = HybridFusionRegressor(fp_dim=fp_dim, node_dim=node_dim, hidden=64)
    pred = model(x_fp, x_nodes, adj)

    assert pred.shape == (batch,)
    print("exercise 79 smoke check passed")
