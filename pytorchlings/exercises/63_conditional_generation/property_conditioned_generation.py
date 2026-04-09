"""Exercise 63: property-conditioned generation with a tiny GRU decoder.

Goal:
- condition generation on a target scalar property (toy solubility)
- wire embedding + recurrent decoder + teacher forcing loss
"""
from __future__ import annotations

import torch
from torch import nn

TOKENS = ["<pad>", "<bos>", "<eos>", "C", "O", "N"]
VOCAB = {t: i for i, t in enumerate(TOKENS)}


def encode(seq: str, max_len: int = 10) -> torch.Tensor:
    ids = [VOCAB["<bos>"]] + [VOCAB[c] for c in seq] + [VOCAB["<eos>"]]
    ids = ids[:max_len] + [VOCAB["<pad>"]] * max(0, max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class ConditionalGenerator(nn.Module):
    def __init__(self, d_model: int = 32):
        super().__init__()
        self.token_emb = nn.Embedding(len(TOKENS), d_model)
        self.prop_proj = nn.Linear(1, d_model)
        self.rnn = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.head = nn.Linear(d_model, len(TOKENS))

    def forward(self, ids: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        tok = self.token_emb(ids)
        # TODO: project scalar prop [B, 1] into d_model and broadcast over sequence length.
        cond = torch.zeros_like(tok)
        # TODO: add conditioning vector to token embeddings before feeding GRU.
        h, _ = self.rnn(tok)
        return self.head(h)


def teacher_forcing_loss(model: nn.Module, ids: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
    """Next-token loss using shifted inputs and targets."""
    x_in = ids[:, :-1]
    y_target = ids[:, 1:]
    logits = model(x_in, prop)
    # TODO: compute CE loss ignoring pad tokens.
    return torch.tensor(0.0)


if __name__ == "__main__":
    torch.manual_seed(0)

    seqs = ["CO", "CCO", "NCO", "CCN"]
    # toy property: higher value for more oxygen atoms
    props = torch.tensor([[1.0], [0.7], [0.9], [0.3]])
    ids = torch.stack([encode(s) for s in seqs])

    model = ConditionalGenerator(d_model=24)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for step in range(5):
        loss = teacher_forcing_loss(model, ids, props)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step={step} loss={float(loss.item()):.4f}")
