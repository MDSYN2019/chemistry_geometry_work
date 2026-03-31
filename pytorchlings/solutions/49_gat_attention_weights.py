"""Solution 49: return attention weights from a GAT layer."""
import torch
from torch import nn
from torch_geometric.nn import GATConv


class InspectableGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=2, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # call GATConv with return_attention_weights=True.
        out, (_, alpha) = self.gat(x, edge_index, return_attention_weights=True)
        return out, alpha
