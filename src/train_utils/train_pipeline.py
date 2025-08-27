# src/train_utils/train_pipeline.py

"""
Entry point for training a segmentation model using PyTorch Lightning.

- Instantiates datasets from predefined train/val folders.
- Builds dataloaders.
- Constructs the segmentation model.
- Initializes logging and callbacks.
- Runs training with the Lightning Trainer.

Args:
    cfg (DictConfig): Hydra configuration object.
"""

from pathlib import Path
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import Logger

from src.utils.seed import set_seed, seed_worker
from src.train_utils.gpu_utils import select_available_gpus, is_rank_zero_worker, is_launcher
from src.train_utils.models.lit_segmentation import LitSegmentation
from src.train_utils.data.dataset import FieldDataset
from src.train_utils.data.collate import get_batch_collate_fn
from src.train_utils.augmentation_visualizer import vis_augmentation_batch
from src.train_utils.dataloader_visualizer import vis_dataloader_batch


def run_train_pipeline(cfg: DictConfig) -> None:
    """Full training pipeline (datasets → loaders → model → trainer.fit)."""
    # === 0) Seed ===
    set_seed(cfg.train.seed)

    # === 1) Datasets ===
    train_ds = FieldDataset(cfg, mode="train")
    val_ds   = FieldDataset(cfg, mode="val")

    # === 2) DataLoaders ===
    generator = torch.Generator()
    generator.manual_seed(cfg.train.seed)

    train_collate_fn = get_batch_collate_fn(cfg.augment.train.batch)
    val_collate_fn   = get_batch_collate_fn(cfg.augment.val.batch)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=train_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=val_collate_fn,  # no batch mixing on val
    )

    # === 3) Model ===
    # Only matters on Ampere+ (A100, A30, etc.)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        # Accuracy-first w/ Tensor Cores (good default)
        torch.set_float32_matmul_precision("high")
        # Optional: also allow TF32 in cuDNN/convolutions
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = LitSegmentation(cfg)

    # === 4) Loggers ===
    loggers: List[Logger] = [hydra.utils.instantiate(lcfg) for lcfg in cfg.train.logger] if is_rank_zero_worker(cfg) else []

    # Optional: visualize dataloader batches
    if getattr(cfg.train, "dataloader_visualizer", False) and cfg.train.dataloader_visualizer.enabled and is_launcher(cfg):
        vis_dataloader_batch(cfg, logger_cfgs=cfg.train.logger)      

    # === 5) Callbacks ===
    checkpoint_path = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch:02d}-{step}-{val_loss:.2f}",
        monitor=cfg.train.checkpoint.monitor,
        mode=cfg.train.checkpoint.mode,
        save_top_k=cfg.train.checkpoint.save_top_k,
        save_last=cfg.train.checkpoint.save_last,
    )
    earlystop_cb = EarlyStopping(
        monitor=cfg.train.early_stop.monitor,
        mode=cfg.train.early_stop.mode,
        patience=cfg.train.early_stop.patience,
    )

    # # === 6) Devices ===
    # if cfg.train.use_multi_gpu:
    #     gpu_ids = select_available_gpus(
    #         max_gpus=min(cfg.train.num_gpus, 3), exclude_ids=[0]
    #     )
    #     devices = gpu_ids
    # else:
    #     devices = 1

    # === 7) Trainer ===
    trainer = Trainer(
        accelerator=cfg.train.trainer.accelerator,
        # devices=devices,
        precision=cfg.train.trainer.precision,
        max_epochs=cfg.train.max_epochs,
        deterministic=cfg.train.trainer.deterministic,
        logger=loggers,
        callbacks=[checkpoint_cb, earlystop_cb],
        default_root_dir=str(Path(cfg.paths.project_train_dir)),
    )

    # === 8) Train ===
    trainer.fit(model, train_loader, val_loader)
    if trainer.is_global_zero:
        print("Training complete.")

    # === 9) Export best weights as .pth ===
    best_ckpt_path = checkpoint_cb.best_model_path
    if best_ckpt_path:
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model_weights = best_ckpt["state_dict"]

        ckpt_filename = Path(best_ckpt_path).stem + ".pth"
        export_path = Path(HydraConfig.get().runtime.output_dir) / "model"
        export_path.mkdir(parents=True, exist_ok=True)
        torch.save(model_weights, export_path / ckpt_filename)
        if trainer.is_global_zero:
            print(f"Best model weights saved to: {export_path / ckpt_filename}")
    else:
        if trainer.is_global_zero:
            print("No best checkpoint found. Skipping .pth export.")
