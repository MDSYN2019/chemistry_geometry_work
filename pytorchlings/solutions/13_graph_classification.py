import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool


class GraphClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.gin = GINConv(nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.gin(x, edge_index)
        g = global_mean_pool(h, batch)
        return self.head(g)
