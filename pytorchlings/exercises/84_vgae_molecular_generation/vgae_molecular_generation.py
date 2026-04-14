"""Exercise 84: VGAE scaffold for molecular-style graph generation.

Goal:
- implement variational graph autoencoder encode/decode path
- train with reconstruction + KL terms
- sample latent vectors and decode candidate graph edges
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data


class TinyVGAEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mu_head(h), self.logvar_head(h)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def decode_inner_product(z: torch.Tensor) -> torch.Tensor:
    return z @ z.T


def make_toy_molecule_graph(n: int = 20, d: int = 12) -> Data:
    torch.manual_seed(84)
    x = torch.randn(n, d)
    src = torch.randint(0, n, (n * 3,))
    dst = torch.randint(0, n, (n * 3,))
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index)


if __name__ == "__main__":
    data = make_toy_molecule_graph()
    enc = TinyVGAEEncoder(in_dim=data.x.size(1), hidden=32, latent=16)
    mu, logvar = enc(data.x)
    z = reparameterize(mu, logvar)
    logits = decode_inner_product(z)

    assert mu.shape == logvar.shape == (data.num_nodes, 16)
    assert logits.shape == (data.num_nodes, data.num_nodes)

    # TODO: add BCE reconstruction on positive/negative edges + KL divergence training loop.
    print("exercise 84 scaffold ready")
