"""Exercise 56: R-GCN for multi-relation graphs."""
import torch
from torch import nn
from torch_geometric.nn import RGCNConv


class RGCNNodeModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_relations: int):
        super().__init__()
        self.rgcn1 = RGCNConv(in_dim, hidden, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden, out_dim, num_relations=num_relations)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        # TODO: run two relational convolutions with relu between.
        h = self.rgcn1(x, edge_index, edge_type).relu()
        return self.rgcn2(h, edge_index, edge_type)
