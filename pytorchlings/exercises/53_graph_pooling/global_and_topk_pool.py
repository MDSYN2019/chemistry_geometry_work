"""Exercise 53: graph pooling with TopK + global mean pooling."""
import torch
from torch import nn
from torch_geometric.nn import GraphConv, TopKPooling, global_mean_pool


class PooledGraphNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.conv = GraphConv(in_dim, hidden)
        self.pool = TopKPooling(hidden, ratio=0.5)
        self.cls = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # TODO: conv -> relu -> topk pool -> global mean pool -> linear.
        x = self.conv(x, edge_index).relu()
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        g = global_mean_pool(x, batch)
        return self.cls(g)
