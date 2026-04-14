"""Exercise 80: Laplacian embedding + spectral clustering.

Goal:
- build graph Laplacians from adjacency
- extract non-trivial eigenvectors as node embeddings
- cluster embeddings and compare to known communities
"""

from __future__ import annotations

import torch


def ring_with_chord_adjacency(n: int = 8) -> torch.Tensor:
    a = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        a[i, (i + 1) % n] = 1.0
        a[(i + 1) % n, i] = 1.0
    a[0, 4] = 1.0
    a[4, 0] = 1.0
    return a


def laplacian_unnormalized(a: torch.Tensor) -> torch.Tensor:
    d = torch.diag(a.sum(dim=1))
    return d - a


def laplacian_normalized(a: torch.Tensor) -> torch.Tensor:
    d = a.sum(dim=1)
    d_inv_sqrt = torch.where(d > 0, d.rsqrt(), torch.zeros_like(d))
    d_inv_sqrt_mat = torch.diag(d_inv_sqrt)
    i = torch.eye(a.size(0), dtype=a.dtype)
    return i - d_inv_sqrt_mat @ a @ d_inv_sqrt_mat


def top_nontrivial_evecs(l: torch.Tensor, k: int = 2) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(l)
    idx = torch.argsort(vals)
    # skip first eigenvector (trivial component)
    keep = idx[1 : 1 + k]
    return vecs[:, keep]


if __name__ == "__main__":
    a = ring_with_chord_adjacency(8)
    l = laplacian_unnormalized(a)
    l_norm = laplacian_normalized(a)
    emb = top_nontrivial_evecs(l_norm, k=2)

    assert a.shape == (8, 8)
    assert l.shape == (8, 8) and l_norm.shape == (8, 8)
    assert emb.shape == (8, 2)

    # TODO: add k-means over `emb` and report cluster purity/NMI against labels.
    print("exercise 80 scaffold ready")
