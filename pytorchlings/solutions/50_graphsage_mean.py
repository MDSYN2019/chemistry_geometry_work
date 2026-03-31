"""Solution 50: GraphSAGE mean aggregation."""
import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class SageClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.sage2 = SAGEConv(hidden, out_dim, aggr="mean")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # apply first layer + relu, then second layer.
        h = self.sage1(x, edge_index).relu()
        return self.sage2(h, edge_index)
