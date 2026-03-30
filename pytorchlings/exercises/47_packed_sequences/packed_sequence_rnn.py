"""Exercise 47: variable-length RNN with pack/pad utilities."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def run_packed_gru(x_padded: torch.Tensor, lengths: torch.Tensor, hidden_size: int = 8) -> torch.Tensor:
    """Run a GRU over padded batch [B, T, F] and return unpacked outputs [B, T, H]."""
    gru = nn.GRU(input_size=x_padded.size(-1), hidden_size=hidden_size, batch_first=True)

    # TODO: pack with enforce_sorted=False, run GRU, then unpack with total_length
    out, _ = gru(x_padded)
    return out
