"""Solution 79: hybrid fingerprint + graph architecture."""

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
        self.gate = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x_nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh_mean = (adj @ x_nodes) / deg
        h = torch.relu(self.self_lin(x_nodes) + self.neigh_lin(neigh_mean))

        score = torch.sigmoid(self.gate(h))
        graph_repr = (score * h).sum(dim=1) / score.sum(dim=1).clamp_min(1e-6)
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
    adj = (adj + torch.eye(n_nodes).unsqueeze(0)).clamp(max=1.0)

    model = HybridFusionRegressor(fp_dim=fp_dim, node_dim=node_dim, hidden=64)
    pred = model(x_fp, x_nodes, adj)

    assert pred.shape == (batch,)
    print("solution 79 smoke check passed")
