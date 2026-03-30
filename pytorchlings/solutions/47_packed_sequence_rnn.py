import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def run_packed_gru(x_padded: torch.Tensor, lengths: torch.Tensor, hidden_size: int = 8) -> torch.Tensor:
    gru = nn.GRU(input_size=x_padded.size(-1), hidden_size=hidden_size, batch_first=True)
    packed = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_out, _ = gru(packed)
    out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x_padded.size(1))
    return out
