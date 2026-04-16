"""Exercise 87: practical Lightning callbacks for stable training.

Goal:
- Configure EarlyStopping + ModelCheckpoint around `val_loss`
- Attach LearningRateMonitor for optimizer diagnostics
- Build a deterministic Trainer for reproducible debugging
"""
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


def build_callbacks() -> list:
    # TODO: create EarlyStopping monitoring "val_loss" with mode="min" and patience=3
    # TODO: create ModelCheckpoint monitoring "val_loss" with mode="min" and save_top_k=1
    # TODO: create LearningRateMonitor with logging_interval="epoch"
    # TODO: return callbacks list in the order [early_stop, checkpoint, lr_monitor]
    raise NotImplementedError


def build_trainer(max_epochs: int = 10) -> L.Trainer:
    callbacks = build_callbacks()
    # TODO: return a Trainer that is deterministic, uses callbacks above,
    # TODO: disables progress bar, and disables default logger
    raise NotImplementedError
