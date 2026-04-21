"""Exercise 94: multimodal adapter - fuse text and molecular descriptors.

Goal:
- bridge from pure LLM text modeling to chemistry-aware multimodal modeling
- concatenate text embedding with numeric molecular descriptors
- predict a property or uncertainty-aware score
"""

from __future__ import annotations

import torch
from torch import nn


class ChemMultimodalAdapter(nn.Module):
    def __init__(self, vocab_size: int, text_dim: int = 64, desc_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, text_dim)
        self.desc_proj = nn.Linear(desc_dim, text_dim)
        self.fusion = nn.Sequential(
            nn.Linear(text_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, token_ids: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T], descriptors: [B, D]
        tok = self.emb(token_ids)

        # TODO: replace simple mean pooling with masked pooling over non-pad tokens.
        text_vec = tok.mean(dim=1)
        desc_vec = self.desc_proj(descriptors)

        fused = torch.cat([text_vec, desc_vec], dim=-1)
        return self.fusion(fused).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(94)
    bsz, t, vocab, d = 5, 12, 40, 16

    x = torch.randint(0, vocab, (bsz, t))
    descriptors = torch.randn(bsz, d)

    model = ChemMultimodalAdapter(vocab_size=vocab, desc_dim=d)
    y = model(x, descriptors)

    assert y.shape == (bsz,)
    print("exercise 94 scaffold ready")
