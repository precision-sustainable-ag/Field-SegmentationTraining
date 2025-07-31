# src/utils/augmentation_visualizer.py

"""
Augmentation Visualizer
----------------------

Sample a few images/masks from the training DataLoader, apply Albumentations ReplayCompose
transforms, and generate a side-by-side comparison grid showing:

  Row 1: Original Image | Augmented Image | Image Legend
  Row 2: Original Mask  | Augmented Mask  | Mask Legend

Legends list only the names of the transforms actually applied to image vs. mask.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage, ToTensor
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb

from src.data.augment import get_noop_transform


def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return ''.join(word.capitalize() for word in name.split('_'))


def render_legend(
    replay: dict,
    height: int,
    full_width: int,
    mask_mode: bool,
    mask_names: set[str],
    font_divisor: int,
    min_font_size: int,
) -> torch.Tensor:
    """
    Render a vertical legend panel listing applied transform names.

    Args:
        replay: Albumentations ReplayCompose output dict.
        height: Panel height in pixels.
        full_width: Desired panel width (matches image width).
        mask_mode: Show only spatial transforms when True.
        mask_names: Set of PascalCase names of spatial transforms.

    Returns:
        RGB tensor (3, height, full_width) with white background.
    """
    # Determine font size relative to panel height
    font_size   = max(min_font_size, height // font_divisor)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Collect names of transforms actually applied
    labels: list[str] = []
    for rec in replay.get("transforms", []):
        if not rec.get("applied", False):
            continue
        name = rec.get("__class_fullname__", "").split('.')[-1]
        if name in {"ToTensorV2", "ReplayCompose"}:
            continue
        if mask_mode and name not in mask_names:
            continue
        labels.append(name)
    if not labels:
        labels = ["No augmentations applied"]

    # Panel width: half the image width, minimum 100px
    legend_width = max(full_width // 2, 100)
    panel = Image.new("RGB", (legend_width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    y_offset = 5
    for label in labels:
        draw.text((5, y_offset), label, fill="black", font=font)
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_h = bbox[3] - bbox[1]
        else:
            text_h = font_size
        y_offset += text_h + 4

    legend_tensor = ToTensor()(panel)
    pad_w = full_width - legend_width
    if pad_w > 0:
        legend_tensor = F.pad(legend_tensor, (0, pad_w, 0, 0), value=1.0)
    return legend_tensor


def vis_augmentation_batch(
    train_loader: torch.utils.data.DataLoader,
    logger_cfgs: list,
    num_samples: int = 4
) -> None:
    """
    Generate and save a combined comparison grid:

      Row 1: Original Image | Augmented Image | Image Legend
      Row 2: Original Mask  | Augmented Mask  | Mask Legend

    Saves to <hydra_run_dir>/image_logs/aug_comparison_full.png
    and logs via each configured logger (e.g. WandB).

    Args:
        train_loader: DataLoader with dataset using ReplayCompose.
        logger_cfgs: List of OmegaConf logger configs.
        num_samples: How many random samples to visualize.
    """
    # Only run on global rank 0 when distributed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    dataset = train_loader.dataset
    leg_cfg = dataset.cfg_aug.augmentation_logger.legend
    font_div  = leg_cfg.font_divisor
    min_fs    = leg_cfg.min_font_size
    total = len(dataset)
    if total == 0:
        return

    # Derive mask-applicable transform names from augment config
    spatial_cfg = dataset.cfg_aug.train.spatial
    mask_names = {
        snake_to_pascal(key)
        for key, spec in spatial_cfg.items()
        if key != "enable" and spec.get("enable", False)
    }

    # Randomly sample indices
    count = min(num_samples, total)
    indices = torch.randperm(total)[:count].tolist()

    cells: list[torch.Tensor] = []
    for idx in indices:
        # Load original image and mask as numpy arrays
        img_np = np.array(Image.open(dataset.images[idx]).convert("RGB"))
        mask_np = np.array(Image.open(dataset.masks[idx]).convert("L"))

        # Apply augmentations (ReplayCompose)
        aug_out = dataset.transform(image=img_np, mask=mask_np)
        aug_img = aug_out["image"].float() / 255.0
        aug_mask = aug_out["mask"].unsqueeze(0).float() / 255.0
        aug_mask = aug_mask.repeat(3, 1, 1)
        replay = aug_out.get("replay", {})

        # Match original dimensions to augmented
        _, H, W = aug_img.shape
        orig_out = get_noop_transform()(image=img_np, mask=mask_np)
        orig_img = orig_out["image"].float() / 255.0
        orig_img = F.interpolate(orig_img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        orig_mask = orig_out["mask"].unsqueeze(0).float() / 255.0
        orig_mask = orig_mask.repeat(3, 1, 1)
        orig_mask = F.interpolate(orig_mask.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)

        # Render legends for image and mask
        img_leg  = render_legend(replay, height=H, full_width=W, mask_mode=False, mask_names=mask_names, font_divisor=font_div, min_font_size=min_fs)
        mask_leg = render_legend(replay, height=H, full_width=W, mask_mode=True,  mask_names=mask_names, font_divisor=font_div, min_font_size=min_fs)

        # Append image row then mask row
        cells.extend([orig_img, aug_img, img_leg])
        cells.extend([orig_mask, aug_mask, mask_leg])

    # Create grid with 3 columns
    grid = make_grid(cells, nrow=3, padding=4)

    # Save and log
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir = run_dir / "image_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "aug_comparison_full.png"
    save_image(grid, str(out_file))
    print(f"Saved comparison grid to: {out_file}")

    for lg_cfg in logger_cfgs:
        logger = hydra.utils.instantiate(lg_cfg)
        exp = getattr(logger, "experiment", None)
        if exp and hasattr(exp, "log"):
            exp.log({"train/aug_comparison_full": [wandb.Image(str(out_file))]})
