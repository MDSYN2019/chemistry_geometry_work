"""Exercise 52: spectral GNN with Chebyshev convolution."""
import torch
from torch import nn
from torch_geometric.nn import ChebConv


class SpectralChebNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, k: int = 3):
        super().__init__()
        self.cheb1 = ChebConv(in_dim, hidden, K=k)
        self.cheb2 = ChebConv(hidden, out_dim, K=k)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # TODO: apply two ChebConv layers with relu between.
        x = self.cheb1(x, edge_index).relu()
        return self.cheb2(x, edge_index)
