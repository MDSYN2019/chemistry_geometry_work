"""Exercise 58: graph autoencoder reconstruction."""
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GraphAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int):
        super().__init__()
        self.enc1 = GCNConv(in_dim, hidden)
        self.enc2 = GCNConv(hidden, latent)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # TODO: two-layer GCN encoder with relu in between.
        return self.enc2(self.enc1(x, edge_index).relu(), edge_index)

    def decode_adj(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: inner-product decoder with sigmoid.
        return torch.sigmoid(z @ z.t())
