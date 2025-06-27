# src/train.py

import sys
from pathlib import Path

# Add project root to sys.path so that `src` becomes importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import List
from pathlib import Path

import hydra
import hydra.utils
from src.utils.gpu_utils import select_available_gpus
from src.utils.seed import set_seed, seed_worker
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import Logger

from src.models.lit_segmentation import LitSegmentation
from src.data.dataset import FieldDataset


@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
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
    # === 0. Seed === 
    # Set seed before anything else
    set_seed(cfg.train.seed)

    # === 1. Datasets ===
    train_ds = FieldDataset(cfg, mode="train")
    val_ds = FieldDataset(cfg, mode="val")

    # === 2. DataLoaders ===
    generator = torch.Generator()
    generator.manual_seed(cfg.train.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
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
        save_last=cfg.train.checkpoint.save_last,
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
    if trainer.is_global_zero:
        print("Training complete.")

    # === 8. Save best model weights as .pth ===
    best_ckpt_path = checkpoint_cb.best_model_path
    if best_ckpt_path:
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model_weights = best_ckpt["state_dict"]

        # Use checkpoint filename (e.g., 'epoch=2-step=100.ckpt') → 'epoch=2-step=100.pth'
        ckpt_filename = Path(best_ckpt_path).stem + ".pth"

        # Create model export path
        export_path = Path(cfg.paths.project_train_dir) / "model"
        export_path.mkdir(parents=True, exist_ok=True)
        torch.save(model_weights, export_path / ckpt_filename)
        if trainer.is_global_zero:
            print(f"Best model weights saved to: {export_path / ckpt_filename}")
    else:
        if trainer.is_global_zero:
            print("No best checkpoint found. Skipping .pth export.")


if __name__ == "__main__":
    train()
