"""Exercise 26: implement a minimal LightningDataModule."""
import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class ToyRegressionDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        # TODO: create a deterministic synthetic regression dataset
        # TODO: split into train and val TensorDataset objects and assign to self.train_ds/self.val_ds
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        # TODO: return train DataLoader with shuffle=True
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        # TODO: return val DataLoader with shuffle=False
        raise NotImplementedError
