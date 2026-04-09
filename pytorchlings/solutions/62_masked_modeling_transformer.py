from __future__ import annotations

import random

import torch
from torch import nn

TOKENS = ["<pad>", "<mask>", "C", "O", "N", "H", "=", "#", "(", ")"]
VOCAB = {t: i for i, t in enumerate(TOKENS)}


def encode(seq: str, max_len: int = 12) -> torch.Tensor:
    ids = [VOCAB[c] for c in seq]
    ids = ids[:max_len] + [VOCAB["<pad>"]] * max(0, max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def mask_inputs(batch_ids: torch.Tensor, mask_prob: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    masked = batch_ids.clone()
    labels = torch.full_like(batch_ids, fill_value=-100)
    for i in range(batch_ids.size(0)):
        for j in range(batch_ids.size(1)):
            tok = int(batch_ids[i, j].item())
            if tok == VOCAB["<pad>"]:
                continue
            if random.random() < mask_prob:
                labels[i, j] = tok
                masked[i, j] = VOCAB["<mask>"]
    return masked, labels


class MaskedChemTransformer(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4, max_len: int = 12):
        super().__init__()
        self.tok_emb = nn.Embedding(len(TOKENS), d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=2
        )
        self.out = nn.Linear(d_model, len(TOKENS))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        h = self.encoder(x)
        return self.out(h)


def run_step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: torch.Tensor) -> float:
    masked, labels = mask_inputs(batch)
    logits = model(masked)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
