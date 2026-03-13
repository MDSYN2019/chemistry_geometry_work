"""Exercise 17: custom dataset with transforms."""
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, n: int = 50):
        self.x = torch.arange(n, dtype=torch.float32)
        self.y = 3 * self.x - 2

    def __len__(self) -> int:
        # TODO: return dataset length
        return 0

    def __getitem__(self, idx: int):
        # TODO: return dict with keys "x" and "y" as tensors shaped [1]
        return {"x": self.x[idx], "y": self.y[idx]}
