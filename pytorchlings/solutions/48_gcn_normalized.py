"""Solution 48: GCN with explicit normalized adjacency."""
import torch
from torch import nn


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute H = D^{-1/2}(A+I)D^{-1/2} X W."""
        num_nodes = adj.size(0)
        # add self-loops by adding identity to adj.
        a_hat = adj
        # compute degree vector and d_inv_sqrt.
        deg = a_hat.sum(dim=1)
        d_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        # build normalized adjacency with outer-product trick.
        a_norm = a_hat * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
        return a_norm @ self.lin(x)
