# src/train.py

from omegaconf import DictConfig
import hydra
import hydra.utils
import torch
from torch.utils.data import random_split, DataLoader
from models.lit_segmentation import LitSegmentation
from data.dataset import FieldDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

@hydra.main(config_path="../conf", config_name="config")
def train_entry(cfg: DictConfig):
    """
    Entry point for training mode.
    Usage: python train.py +pipeline.mode=train
    """
    # 1) Build dataset + splits
    full = FieldDataset(cfg, mode="train")
    n_train = int(len(full) * cfg.train.train_val_split)
    n_val   = len(full) - n_train
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.train.seed))

    # 2) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False,
                              num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)

    # 3) Model
    model = LitSegmentation(cfg)

    # 4) Logger
    # Dynamically instantiate all configured loggers
    loggers = [hydra.utils.instantiate(lcfg) for lcfg in cfg.train.logger]

    # 5) Callbacks
    ckpt_conf = cfg.train.checkpoint
    checkpoint_cb = ModelCheckpoint(
        monitor=ckpt_conf.monitor,
        mode=ckpt_conf.mode,
        save_top_k=ckpt_conf.save_top_k,
    )
    es_conf = cfg.train.early_stop
    earlystop_cb = EarlyStopping(
        monitor=es_conf.monitor,
        mode=es_conf.mode,
        patience=es_conf.patience
    )

    # 6) Trainer
    trainer = Trainer(
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        precision=cfg.train.trainer.precision,
        max_epochs=cfg.train.max_epochs,
        deterministic=cfg.train.trainer.deterministic,
        logger=loggers,
        callbacks=[checkpoint_cb, earlystop_cb],
        default_root_dir=cfg.paths.project_train_dir
    )

    # 7) Fit
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train_entry()
