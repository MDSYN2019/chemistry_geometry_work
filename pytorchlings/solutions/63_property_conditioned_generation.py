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
        cond = self.prop_proj(prop).unsqueeze(1).expand(-1, ids.size(1), -1)
        h, _ = self.rnn(tok + cond)
        return self.head(h)


def teacher_forcing_loss(model: nn.Module, ids: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
    x_in = ids[:, :-1]
    y_target = ids[:, 1:]
    logits = model(x_in, prop)
    return nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])(logits.reshape(-1, logits.size(-1)), y_target.reshape(-1))
