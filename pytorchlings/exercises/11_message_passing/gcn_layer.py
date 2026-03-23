"""Exercise 11: one GCN layer forward."""
import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class OneLayerGCN(nn.Module):
    def __init__(self, in_channels: int = 8, out_channels: int = 2):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index) # the nodes and the edge index 
        x = x.relu()        
        # TODO: apply conv and relu
        return x# self.conv(x, edge_index)

class TwoLayerGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x
