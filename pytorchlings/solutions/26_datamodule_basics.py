import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class ToyRegressionDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(256, 4, generator=generator)
        noise = 0.05 * torch.randn(256, 1, generator=generator)
        y = (x[:, :1] * 0.8) + (x[:, 1:2] * -0.4) + noise

        self.train_ds = TensorDataset(x[:200], y[:200])
        self.val_ds = TensorDataset(x[200:], y[200:])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
