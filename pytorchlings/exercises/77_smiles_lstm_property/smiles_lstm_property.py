"""Exercise 77: SMILES-style sequence encoder with LSTM for property prediction.

Goal:
- tokenize toy chemistry strings
- encode variable-length sequences with an LSTM
"""

import torch
from torch import nn


class SmilesLSTMRegressor(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(token_ids)
        h, _ = self.lstm(e)

        # TODO: replace last-token heuristic with packed sequence lengths
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)


def encode_strings(strings: list[str], stoi: dict[str, int], pad_to: int) -> torch.Tensor:
    ids = []
    for s in strings:
        row = [stoi.get(ch, stoi["?"]) for ch in s][:pad_to]
        row += [0] * (pad_to - len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long)


if __name__ == "__main__":
    torch.manual_seed(77)
    vocab = ["<pad>", "C", "N", "O", "(", ")", "=", "#", "[", "]", "?", "1"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    smiles = ["CCO", "N#N", "C(=O)O", "CCN", "C1CC1"]
    x = encode_strings(smiles, stoi, pad_to=10)

    model = SmilesLSTMRegressor(vocab_size=len(vocab))
    pred = model(x)

    assert pred.shape == (len(smiles),)
    print("exercise 77 smoke check passed")
