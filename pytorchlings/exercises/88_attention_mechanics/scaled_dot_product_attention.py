"""Exercise 88: understand scaled dot-product attention from first principles."""
import math

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention output and attention probabilities.

    Expected shapes:
      q: (batch, heads, q_len, d_k)
      k: (batch, heads, k_len, d_k)
      v: (batch, heads, k_len, d_v)
      mask: optional broadcastable mask where 0 means "blocked"

    Returns:
      output: (batch, heads, q_len, d_v)
      probs: (batch, heads, q_len, k_len)
    """
    # TODO: compute raw attention scores q @ k^T
    # TODO: scale by sqrt(d_k)
    # TODO: if mask is provided, fill masked positions with a large negative value
    # TODO: apply softmax over the key dimension
    # TODO: multiply probabilities by v to get output
    raise NotImplementedError


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a lower-triangular causal mask of shape (1, 1, seq_len, seq_len)."""
    # TODO: build lower triangular mask with ones on/under diagonal
    raise NotImplementedError
