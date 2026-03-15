"""Exercise 25: build a minimal LightningModule training step."""
import lightning as L
import torch
from torch import nn


class LitRegressor(L.LightningModule):
    def __init__(self, in_features: int = 4, hidden_dim: int = 16, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        # TODO: define a tiny feed-forward network and assign it to self.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return model predictions
        raise NotImplementedError

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # TODO: unpack batch, compute mse loss, log "train_loss", and return loss
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # TODO: return Adam optimizer configured with self.hparams.lr
        raise NotImplementedError
