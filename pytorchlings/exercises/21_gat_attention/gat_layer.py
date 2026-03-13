"""Exercise 21: GAT layer forward."""
import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=2, concat=True)
        self.gat2 = GATConv(hidden * 2, out_dim, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # TODO: apply ELU after first GAT layer
        h = self.gat1(x, edge_index)
        return self.gat2(h, edge_index)
