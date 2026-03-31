"""Exercise 59: diffusion-based GNN with APPNP propagation."""
import torch
from torch import nn
from torch_geometric.nn import APPNP


class DiffusionAPPNP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, k: int = 10, alpha: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.propagate = APPNP(K=k, alpha=alpha)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # TODO: compute logits with MLP then diffuse with APPNP.
        logits = self.mlp(x)
        return self.propagate(logits, edge_index)
