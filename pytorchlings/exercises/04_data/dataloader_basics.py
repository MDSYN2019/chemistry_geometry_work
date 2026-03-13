"""Exercise 04: DataLoader basics."""
import torch
from torch.utils.data import TensorDataset, DataLoader


def build_loader(batch_size: int = 4) -> DataLoader:
    x = torch.arange(0, 20, dtype=torch.float32).unsqueeze(1)
    y = 2 * x + 1
    ds = TensorDataset(x, y)
    # TODO: enable shuffling
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
