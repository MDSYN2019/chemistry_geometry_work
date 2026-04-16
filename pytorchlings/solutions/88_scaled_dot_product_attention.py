"""Solution 88: scaled dot-product attention from first principles."""
import math

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-2, -1))
    d_k = q.size(-1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    return output, probs


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    lower = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return lower.view(1, 1, seq_len, seq_len)
