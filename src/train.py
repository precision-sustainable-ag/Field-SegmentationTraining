# src/train.py

from typing import List
from pathlib import Path

import hydra
import hydra.utils
from utils.gpu_utils import select_available_gpus
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import Logger

from models.lit_segmentation import LitSegmentation
from data.dataset import FieldDataset


@hydra.main(config_path="../conf", config_name="config")
def train_entry(cfg: DictConfig) -> None:
    """
    Entry point for training a segmentation model using PyTorch Lightning.

    This function:
    - Instantiates datasets from predefined train/val folders.
    - Builds dataloaders.
    - Constructs the segmentation model.
    - Initializes logging and callbacks.
    - Runs training with the Lightning Trainer.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    # === 1. Datasets ===
    train_ds = FieldDataset(cfg, mode="train")
    val_ds = FieldDataset(cfg, mode="val")

    # === 2. DataLoaders ===
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    # === 3. Model ===
    model = LitSegmentation(cfg)

    # === 4. Loggers ===
    # Dynamically instantiate all configured loggers
    loggers: List[Logger] = [hydra.utils.instantiate(lcfg) for lcfg in cfg.train.logger]

    # === 5. Callbacks ===
    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.train.checkpoint.monitor,
        mode=cfg.train.checkpoint.mode,
        save_top_k=cfg.train.checkpoint.save_top_k,
    )
    earlystop_cb = EarlyStopping(
        monitor=cfg.train.early_stop.monitor,
        mode=cfg.train.early_stop.mode,
        patience=cfg.train.early_stop.patience,
    )

    # === 6. Trainer ===
    if cfg.train.use_multi_gpu:
        gpu_ids = select_available_gpus(max_gpus=min(cfg.train.num_gpus, 3), exclude_ids=[0])
        devices = gpu_ids
    else:
        devices = 1

    trainer = Trainer(
        accelerator=cfg.train.trainer.accelerator,
        devices=devices,
        precision=cfg.train.trainer.precision,
        max_epochs=cfg.train.max_epochs,
        deterministic=cfg.train.trainer.deterministic,
        logger=loggers,
        callbacks=[checkpoint_cb, earlystop_cb],
        default_root_dir=str(Path(cfg.paths.project_train_dir))
    )


    # === 7. Train ===
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train_entry()
