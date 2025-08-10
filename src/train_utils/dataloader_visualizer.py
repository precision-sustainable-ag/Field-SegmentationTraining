# src/train_utils/dataloader_visualizer.py

import math
import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
import torchvision
import torch
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from typing import Optional, List
import wandb

from src.train_utils.data.dataset import FieldDataset
from src.train_utils.data.collate import get_batch_collate_fn

def _get_run_dir() -> Path:
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    return Path.cwd()


def vis_dataloader_batch(cfg: DictConfig, logger_cfgs: Optional[List] = None) -> None:
    """
    Visualize one training batch and (optionally) log to W&B/other Lightning loggers.

    Saves:
      image_logs/batch_visualization_image.png
      image_logs/batch_visualization_mask.png

    If `logger_cfgs` is provided, logs:
      - train/dataloader_image_grid  -> wandb.Image
      - train/dataloader_mask_grid   -> wandb.Image
    """
    # Ensure only the main process visualizes
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    # Dataset & loader
    ds = FieldDataset(cfg, mode="train")
    # supports either a simple bool flag or a sub-config with num_samples
    n_samples = getattr(getattr(cfg.train, "dataloader_visualizer", {}), "num_samples", 4)
    n_samples = int(n_samples) if isinstance(n_samples, (int, float, str)) else 4

    if len(ds) == 0:
        print("[vis_dataloader] Dataset is empty; skipping.")
        return

    collate = get_batch_collate_fn(cfg.augment.train.batch)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=min(max(1, n_samples), len(ds)),
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    images, masks = next(iter(loader))  # [B,3,H,W], [B,1,H,W] or [B,3,H,W]
    B = images.size(0)
    nrow = min(4, B) if B <= 8 else math.ceil(math.sqrt(B))

    out_dir = _get_run_dir() / "image_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Images grid
    img_grid = torchvision.utils.make_grid(images, nrow=nrow, padding=4)
    img_path = out_dir / "batch_visualization_image.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title("Batch Images")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    # Masks grid (expand to 3 channels for display)
    if masks.ndim == 4 and masks.size(1) == 1:
        masks_vis = masks.expand(-1, 3, -1, -1)
    elif masks.ndim == 3:
        masks_vis = masks.unsqueeze(1).expand(-1, 3, -1, -1)
    else:
        masks_vis = masks

    mask_grid = torchvision.utils.make_grid(masks_vis, nrow=nrow, padding=4)
    mask_path = out_dir / "batch_visualization_mask.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_grid.permute(1, 2, 0))
    plt.title("Batch Masks")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(mask_path)
    plt.close()

    # Optional: log to configured loggers (e.g., WandbLogger)
    if wandb.run is not None:
        # log to the current run (preferred)
        wandb.log({
            "train/dataloader_image_grid": wandb.Image(str(img_path)),
            "train/dataloader_mask_grid":  wandb.Image(str(mask_path)),
        })
    elif logger_cfgs:
        # fallback: create a run only if none exists
        for lcfg in logger_cfgs:
            logger = hydra.utils.instantiate(lcfg)
            exp = getattr(logger, "experiment", None)
            if exp and hasattr(exp, "log"):
                exp.log({
                    "train/dataloader_image_grid": wandb.Image(str(img_path)),
                    "train/dataloader_mask_grid":  wandb.Image(str(mask_path)),
                })

# --- CLI wrapper for standalone usage ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def _cli_entry(cfg: DictConfig) -> None:
    vis_dataloader_batch(cfg, logger_cfgs=cfg.train.logger)

if __name__ == "__main__":
    _cli_entry()
