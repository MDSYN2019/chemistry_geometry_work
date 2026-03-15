import lightning as L
import torch
from torch import nn


class LitRegressor(L.LightningModule):
    def __init__(self, in_features: int = 4, hidden_dim: int = 16, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
