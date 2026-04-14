"""Exercise 83: self-supervised graph pretraining + fine-tune.

Goal:
- pretrain a node encoder with masked feature reconstruction
- transfer encoder to downstream node classification
- compare scratch vs pretrained initialization
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedFeatureHead(nn.Module):
    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


def make_graph(n: int = 256, d: int = 24) -> Data:
    torch.manual_seed(83)
    x = torch.randn(n, d)
    src = torch.randint(0, n, (n * 4,))
    dst = torch.randint(0, n, (n * 4,))
    y = torch.randint(0, 4, (n,))
    return Data(x=x, edge_index=torch.stack([src, dst], dim=0), y=y)


def random_mask(x: torch.Tensor, p: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(x).lt(p)
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


if __name__ == "__main__":
    data = make_graph()
    encoder = MLPEncoder(in_dim=data.x.size(1), hidden=32)
    head = MaskedFeatureHead(hidden=32, out_dim=data.x.size(1))

    x_masked, mask = random_mask(data.x)
    z = encoder(x_masked)
    x_hat = head(z)

    assert x_hat.shape == data.x.shape
    assert mask.dtype == torch.bool

    # TODO: implement pretrain loop, then downstream fine-tune comparison against scratch.
    print("exercise 83 scaffold ready")
