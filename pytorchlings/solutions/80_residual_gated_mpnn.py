"""Solution 80: residual + gated MPNN stack."""

import torch
from torch import nn


class ResidualMPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.self_lin = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z = self.norm(h)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = (adj @ z) / deg
        update = torch.tanh(self.self_lin(z) + self.neigh_lin(neigh))
        mix = torch.sigmoid(self.gate(z))
        h = mix * update + (1.0 - mix) * h
        return h + self.ff(self.norm(h))


class ResidualGatedMPNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, depth: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResidualMPBlock(hidden) for _ in range(depth)])
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(x_nodes))
        for block in self.blocks:
            h = block(h, adj)
        graph_repr = self.readout(h).mean(dim=1)
        return self.head(graph_repr).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(80)
    batch, n_nodes, node_dim = 6, 20, 32
    x_nodes = torch.randn(batch, n_nodes, node_dim)
    a = torch.randint(0, 2, (batch, n_nodes, n_nodes), dtype=torch.float32)
    adj = torch.triu(a, diagonal=1)
    adj = adj + adj.transpose(-1, -2)
    adj = (adj + torch.eye(n_nodes).unsqueeze(0)).clamp(max=1.0)

    model = ResidualGatedMPNN(in_dim=node_dim, hidden=64, depth=5)
    pred = model(x_nodes, adj)

    assert pred.shape == (batch,)
    assert torch.isfinite(pred).all()
    print("solution 80 smoke check passed")
