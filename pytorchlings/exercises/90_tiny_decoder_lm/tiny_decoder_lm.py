"""Exercise 90: tiny decoder-only LM from scratch for chemistry strings.

Goal:
- implement causal self-attention masking
- train a tiny autoregressive transformer on toy SMILES-like text
- understand the basic training loop behind modern LLMs
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4, layers: int = 2, block_size: int = 16):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos_ids = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos_ids)

        # TODO: create a causal mask so position i cannot attend to j > i.
        # Expected mask shape for nn.TransformerEncoder with batch_first=True: [t, t].
        causal_mask = None

        h = self.decoder(h, mask=causal_mask)
        return self.head(h)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # TODO: flatten [B, T, V] and [B, T] to compute cross entropy.
    return F.cross_entropy(torch.zeros((1, logits.size(-1))), torch.zeros((1,), dtype=torch.long))


if __name__ == "__main__":
    torch.manual_seed(90)

    vocab_size = 32
    x = torch.randint(0, vocab_size, (4, 16))
    y = torch.randint(0, vocab_size, (4, 16))

    model = TinyDecoderLM(vocab_size=vocab_size)
    logits = model(x)
    assert logits.shape == (4, 16, vocab_size)

    loss = lm_loss(logits, y)
    print("loss:", float(loss.item()))
    print("exercise 90 scaffold ready")
