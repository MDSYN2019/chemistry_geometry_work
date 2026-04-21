"""Exercise 89: tokenization + dataset prep for chemistry LLM work.

Goal:
- start from first principles: turn a tiny chemistry corpus into model-ready tensors
- compare simple character tokenization with domain-aware tokenization ideas
- build train/validation windows for autoregressive next-token prediction
"""

from __future__ import annotations

import random

import torch


def build_char_vocab(texts: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Return (stoi, itos) including special tokens.

    Special tokens:
    - <pad>: 0
    - <bos>: 1
    - <eos>: 2
    """
    # TODO: collect unique characters from texts and build stoi/itos.
    # Keep ordering deterministic (e.g., sorted chars).
    stoi = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos


def encode_sequence(seq: str, stoi: dict[str, int]) -> list[int]:
    # TODO: encode with <bos> at start and <eos> at end.
    return [stoi["<bos>"], stoi["<eos>"]]


def build_lm_windows(encoded: list[int], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build next-token windows (x, y) where y is x shifted by one token."""
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    # TODO: create sliding windows of length block_size over encoded tokens.
    if not xs:
        x = torch.zeros((0, block_size), dtype=torch.long)
        y = torch.zeros((0, block_size), dtype=torch.long)
        return x, y
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


if __name__ == "__main__":
    random.seed(89)
    torch.manual_seed(89)

    corpus = ["CCO", "N#N", "C(=O)O", "CCN", "ClCCl", "c1ccccc1"]
    stoi, itos = build_char_vocab(corpus)
    assert stoi["<pad>"] == 0 and itos[0] == "<pad>"

    stream = []
    for s in corpus:
        stream.extend(encode_sequence(s, stoi))

    x, y = build_lm_windows(stream, block_size=8)
    print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
    print("exercise 89 scaffold ready")
