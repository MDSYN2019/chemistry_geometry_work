import torch
from torch import nn
from torch_geometric.nn import GCNConv


class NodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)
        return self.conv2(h, edge_index)
