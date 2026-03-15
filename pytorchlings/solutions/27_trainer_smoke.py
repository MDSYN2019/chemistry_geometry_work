import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


def build_trainer(max_epochs: int = 3) -> L.Trainer:
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    return L.Trainer(
        max_epochs=max_epochs,
        logger=False,
        deterministic=True,
        callbacks=[checkpoint_cb],
        enable_progress_bar=False,
    )


def run_smoke_train(model: L.LightningModule, datamodule: L.LightningDataModule) -> None:
    trainer = build_trainer(max_epochs=2)
    trainer.fit(model=model, datamodule=datamodule)
