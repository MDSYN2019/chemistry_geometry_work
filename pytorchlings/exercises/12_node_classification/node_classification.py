"""Exercise 12: node classification model."""
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class NodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, n_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # TODO: add relu + dropout between conv layers
        h = self.conv1(x, edge_index)
        return self.conv2(h, edge_index)
