import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from configs.defaults import DATA_BATCH_SIZE, NUM_WORKERS, TRAINER_CONFIG
from experiments.runtime import detect_runtime
from models.data_module import AtacamaDataModule


def build_data_module(splits):
    data_module = AtacamaDataModule(
        x_train=splits["train"],
        x_val=splits["val"],
        x_calib=splits["calib"],
        x_test=splits["test"],
        batch_size=DATA_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    data_module.setup()
    return data_module


def build_trainer(output_dir):
    runtime = detect_runtime()
    logger = CSVLogger(save_dir=output_dir, name="lstm_ae_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=TRAINER_CONFIG["max_epochs"],
        accelerator=runtime["accelerator"],
        devices=TRAINER_CONFIG["devices"]
        if TRAINER_CONFIG["devices"] != "auto"
        else runtime["devices"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=TRAINER_CONFIG["log_every_n_steps"],
    )
    return trainer, checkpoint_callback


def train_model(model, data_module, output_dir):
    trainer, checkpoint_callback = build_trainer(output_dir)
    print(f"Starting training. Logs and checkpoints will be saved to: {output_dir}")
    trainer.fit(model, datamodule=data_module)
    print("-" * 50)
    print("Training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path
