"""Exercise 27: configure a Lightning Trainer smoke test."""
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


def build_trainer(max_epochs: int = 3) -> L.Trainer:
    # TODO: create a ModelCheckpoint callback monitoring "val_loss" (mode="min")
    # TODO: return Trainer with logger disabled, checkpointing enabled, and deterministic=True
    raise NotImplementedError


def run_smoke_train(model: L.LightningModule, datamodule: L.LightningDataModule) -> None:
    trainer = build_trainer(max_epochs=2)
    trainer.fit(model=model, datamodule=datamodule)
