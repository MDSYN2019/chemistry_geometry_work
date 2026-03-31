"""Exercise 51: GIN with learnable MLP update."""
import torch
from torch import nn
from torch_geometric.nn import GINConv


class GINNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gin1 = GINConv(mlp)
        self.lin_out = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # TODO: pass through GINConv, relu, then output linear.
        h = self.gin1(x, edge_index).relu()
        return self.lin_out(h)
