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
from typing import Optional, List, Dict
import wandb
import json

from src.train_utils.data.dataset import FieldDataset
from src.train_utils.data.collate import get_batch_collate_fn
from torch.utils.data._utils.collate import default_collate

def _get_run_dir() -> Path:
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    return Path.cwd()

# --- display conversion helpers -----------------------------------------

def _detect_sample_norm_kind(cfg: DictConfig) -> Optional[str]:
    """Return 'image' | 'image_per_channel' | 'dataset_wide' | None based on active norm."""
    
    # 1. Check if dataset-wide normalization is active
    if getattr(cfg.train, "use_data_normalization", False):
        return "dataset_wide"
        
    # 2. Check if sample-specific normalization was applied by Albumentations
    norm_cfg = getattr(cfg.augment.train, "normalization", None)
    if norm_cfg and norm_cfg.get("enable", False):
        kind = str(norm_cfg.get("kind", "")).lower()
        if kind in ("image", "image_per_channel"):
            return kind
            
    return None

def _to_display_batch(imgs: torch.Tensor, norm_kind: Optional[str], dataset_stats: Optional[Dict[str, list]] = None) -> torch.Tensor:
    """
    Convert a batch [B, C, H, W] to [0,1] for visualization.
    - If dataset-wide normalization was applied, mathematically invert it using dataset_stats.
    - If sample-specific normalization was applied by Albumentations,
      do per-image min–max scaling for display.
    - Else, clamp.
    """
    x = imgs.detach()
    
    if norm_kind == "dataset_wide" and dataset_stats is not None:
        # 1. Mathematically invert Z-score normalization
        # Reshape lists to [1, 3, 1, 1] to broadcast across the [B, 3, H, W] batch
        mean = torch.tensor(dataset_stats["mean"], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(dataset_stats["std"], device=x.device).view(1, 3, 1, 1)
        
        x_disp = (x * std) + mean
        return x_disp.clamp(0.0, 1.0)
        
    elif norm_kind in ("image", "image_per_channel"):
        # 2. Dynamic Albumentations: Fallback to global per-image min-max
        B = x.shape[0]
        x_flat = x.view(B, -1)
        mins = x_flat.min(dim=1).values.view(B, 1, 1, 1)
        maxs = x_flat.max(dim=1).values.view(B, 1, 1, 1)
        denom = (maxs - mins).clamp_min(1e-6)
        x_disp = (x - mins) / denom
        return x_disp.clamp(0.0, 1.0)
        
    else:
        # 3. No normalization applied
        return x.clamp(0.0, 1.0)

# -----------------------------------------------------------------------------

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

    # Check the master switch before applying batch-level mixing
    if getattr(cfg.augment.train, "enable", False):
        collate = get_batch_collate_fn(cfg.augment.train.batch)
    else:
        # If train augmentations are false, DO NOT apply any mixing.
        collate = default_collate
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

    # Detect normalization
    norm_kind = _detect_sample_norm_kind(cfg)
    dataset_stats = None

    if norm_kind == "dataset_wide":
        # Read the stats file using your new config path
        stats_path = Path(cfg.paths.project_datastats_dir) / "rgb_mean_std.json"
        try:
            with open(stats_path, "r") as f:
                dataset_stats = json.load(f)
            vis_msg = f"Note: dataset-wide normalization inverted using stats from {stats_path.name}."
        except Exception as e:
            print(f"[vis_dataloader] Warning: Could not read dataset stats: {e}")
            vis_msg = "Note: images clamped (dataset stats failed to load)."
            
    elif norm_kind:
        vis_msg = f"Note: sample-specific normalization ({norm_kind}) detected; min–max rescaled per image."
    else:
        vis_msg = "Note: images clamped for visualization."

    # Images grid (convert to display range first using the new logic)
    images_disp = _to_display_batch(images, norm_kind, dataset_stats)
    img_grid = torchvision.utils.make_grid(images_disp, nrow=nrow, padding=4)
    img_path = out_dir / "batch_visualization_image.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title("Batch Images\n" + vis_msg, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    # Masks grid (expand to 3 channels for display) – masks are already 0/1
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
    plt.title("Batch Masks", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(mask_path)
    plt.close()

    # Console note as well
    print("[vis_dataloader]", vis_msg)

    # Log to W&B / other
    if wandb.run is not None:
        wandb.log({
            "train/dataloader_image_grid": wandb.Image(str(img_path), caption=vis_msg),
            "train/dataloader_mask_grid":  wandb.Image(str(mask_path)),
        })
    elif logger_cfgs:
        for lcfg in logger_cfgs:
            logger = hydra.utils.instantiate(lcfg)
            exp = getattr(logger, "experiment", None)
            if exp and hasattr(exp, "log"):
                exp.log({
                    "train/dataloader_image_grid": wandb.Image(str(img_path), caption=vis_msg),
                    "train/dataloader_mask_grid":  wandb.Image(str(mask_path)),
                })

# --- CLI wrapper for standalone usage ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def _cli_entry(cfg: DictConfig) -> None:
    vis_dataloader_batch(cfg, logger_cfgs=cfg.train.logger)

if __name__ == "__main__":
    _cli_entry()
