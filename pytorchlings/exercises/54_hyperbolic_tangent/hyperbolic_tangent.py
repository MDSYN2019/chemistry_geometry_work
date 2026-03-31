"""Exercise 54: simple hyperbolic-inspired message passing.

We approximate operations in tangent space using atanh/tanh transforms.
"""
import torch
from torch import nn


class HyperbolicLikeLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # TODO: map from Poincare ball to tangent: logmap0(x) ~= atanh(||x||) * x/||x||.
        eps = 1e-6
        norm = x.norm(dim=-1, keepdim=True).clamp(min=eps, max=1 - eps)
        tangent = torch.atanh(norm) * x / norm
        # TODO: aggregate in tangent space and project back with tanh.
        agg = adj @ tangent
        out = self.lin(agg)
        out_norm = out.norm(dim=-1, keepdim=True).clamp(min=eps)
        return torch.tanh(out_norm) * out / out_norm
