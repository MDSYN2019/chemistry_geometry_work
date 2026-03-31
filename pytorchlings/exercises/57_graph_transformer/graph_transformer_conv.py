"""Exercise 57: graph transformer-style attention layer."""
import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class GraphTransformerNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.t1 = TransformerConv(in_dim, hidden, heads=2, concat=True)
        self.t2 = TransformerConv(hidden * 2, out_dim, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # TODO: first transformer conv + relu, then second.
        h = self.t1(x, edge_index).relu()
        return self.t2(h, edge_index)
