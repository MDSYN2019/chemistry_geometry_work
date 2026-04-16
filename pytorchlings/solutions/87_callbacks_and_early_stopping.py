"""Solution 87: practical Lightning callbacks for stable training."""
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


def build_callbacks() -> list:
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [early_stop, checkpoint, lr_monitor]


def build_trainer(max_epochs: int = 10) -> L.Trainer:
    callbacks = build_callbacks()
    return L.Trainer(
        max_epochs=max_epochs,
        deterministic=True,
        callbacks=callbacks,
        enable_progress_bar=False,
        logger=False,
    )
