import torch
from torch import nn
from torch_geometric.nn import GCNConv


class OneLayerGCN(nn.Module):
    def __init__(self, in_channels: int = 8, out_channels: int = 2):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index).relu()
