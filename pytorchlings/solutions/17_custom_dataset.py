import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, n: int = 50):
        self.x = torch.arange(n, dtype=torch.float32)
        self.y = 3 * self.x - 2

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {"x": self.x[idx].unsqueeze(0), "y": self.y[idx].unsqueeze(0)}
