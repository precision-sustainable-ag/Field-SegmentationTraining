# src/train_utils/augmentation_visualizer.py

"""
Augmentation Visualizer
----------------------

Sample a few images/masks from the training DataLoader, apply both:

  1) Your per-sample ReplayCompose Albumentations pipeline, and
  2) Your batch-level mixing (MixUp, CutMix, Mosaic) via the DataLoader’s collate_fn

Then generate a comparison grid:

  Row 1: Original Image | Per-sample Augmented Image | Per-sample Legend | Batch-mixed Image | Batch-Legend
  Row 2: Original Mask  | Per-sample Augmented Mask  | Per-sample Legend | Batch-mixed Mask  | Batch-Legend

Legends list only the transforms actually applied.
"""

import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb

from src.train_utils.data.dataset import FieldDataset
from src.train_utils.data.augment import get_noop_transform
from src.train_utils.data.collate import (
    mixup_collate,
    cutmix_collate,
    mosaic_collate,
    get_batch_collate_fn,
)

# Albumentations class names that (in your new pipeline) affect both image & mask.
# This is used to filter the mask legend so it shows only spatial/mask-safe ops.
SPATIAL_MASK_SAFE = {
    # wrappers
    "Sequential", "OneOf", "SomeOf",
    # geometric & warps
    "HorizontalFlip", "VerticalFlip", "RandomRotate90",
    "Affine", "ShiftScaleRotate", "Perspective",
    "ElasticTransform", "GridDistortion", "OpticalDistortion",
    "RandomScale",
    # spatial sizing
    "RandomCrop", "SmallestMaxSize", "LongestMaxSize", "PadIfNeeded",
    # your custom dual op for masks
    "CoarseDropoutDual",
}

# Things we generally don't want in the legend
LEGEND_SKIP = {"ToTensorV2", "ReplayCompose", "Resize", "PadIfNeeded"}


def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _wrap_label_for_wrapper(cls_name: str, rec: dict) -> str:
    """
    Try to produce a helpful label for wrapper blocks like SomeOf/OneOf/Sequential.
    Albumentations replay dict sometimes contains init-time 'parameters'.
    """
    base = cls_name
    params = rec.get("parameters", {}) or rec.get("params", {})
    # Try to extract N for SomeOf
    if cls_name == "SomeOf":
        n = params.get("n", None)
        if n is not None:
            base = f"SomeOf(n={n})"
    elif cls_name == "Sequential":
        base = "All"
    return base


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
        replay: Albumentations ReplayCompose output dict or custom replay.
        height: Panel height in pixels.
        full_width: Desired panel width (matches image width).
        mask_mode: If True, only include transforms in mask_names.
        mask_names: Set of class names considered mask-safe (for mask legend).
    """
    font_size = max(min_font_size, height // font_divisor)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    labels: list[str] = []
    for rec in replay.get("transforms", []):
        if not rec.get("applied", False):
            continue

        cls_name = rec.get("__class_fullname__", "").split(".")[-1]
        if cls_name in LEGEND_SKIP:
            continue

        # For wrappers, show a concise label
        if cls_name in {"SomeOf", "OneOf", "Sequential"}:
            lbl = _wrap_label_for_wrapper(cls_name, rec)
            # Always show wrappers; they explain selection behavior
            labels.append(lbl)
            continue

        # Only include mask-safe transforms if we're rendering the mask legend
        if mask_mode and cls_name not in mask_names:
            continue

        labels.append(cls_name)

    if not labels:
        labels = ["No augmentations applied"]

    legend_width = max(full_width // 2, 100)
    panel = Image.new("RGB", (legend_width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)
    y = 5
    for lbl in labels:
        draw.text((5, y), lbl, fill="black", font=font)
        bbox = draw.textbbox((0, 0), lbl, font=font)
        text_h = bbox[3] - bbox[1]
        y += text_h + 4

    legend = ToTensor()(panel)
    pad_w = full_width - legend_width
    if pad_w > 0:
        legend = F.pad(legend, (0, pad_w, 0, 0), value=1.0)
    return legend


def vis_augmentation_batch(
    train_loader: torch.utils.data.DataLoader,
    logger_cfgs: list,
    num_samples: int = 4,
) -> None:
    """
    Generate and save a comparison grid including:
      Row 1: Orig Img | Per-sample Aug Img | Per-sample Legend | Batch Mix Img | Batch Legend
      Row 2: Orig Mask| Per-sample Aug Mask| Per-sample Legend | Batch Mix Mask| Batch Legend
    """
    # Only run on rank 0 in distributed mode
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    dataset = train_loader.dataset
    vis_cfg = dataset.cfg_aug.augmentation_visualizer.legend
    font_div, min_fs = vis_cfg.font_divisor, vis_cfg.min_font_size
    total = len(dataset)
    if total == 0:
        return

    # With the new grouped config, just use the known mask-safe class names.
    mask_names = set(SPATIAL_MASK_SAFE)

    batch_cfg = dataset.cfg_aug.train.batch  # batch-level mixing config
    cells: list[torch.Tensor] = []

    for idx in torch.randperm(total)[: min(num_samples, total)].tolist():
        # --- 1) per-sample augment via ReplayCompose ---
        img_np = np.array(Image.open(dataset.images[idx]).convert("RGB"))
        mask_np = np.array(Image.open(dataset.masks[idx]).convert("L"))
        out1 = dataset.transform(image=img_np, mask=mask_np)
        aug_img = out1["image"].float() / 255.0

        tmp_mask = out1["mask"].float() / 255.0  # [H0, W0] or [1,H0,W0]
        if tmp_mask.ndim == 2:
            tmp_mask = tmp_mask.unsqueeze(0)  # [1,H0,W0]
        tmp_mask = tmp_mask.unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H0,W0]
        _, C, H, W = tmp_mask.shape
        aug_mask = F.interpolate(tmp_mask, size=(H, W), mode="nearest").squeeze(0)  # [3,H,W]

        replay1 = out1.get("replay", {"transforms": []})

        # Resize original image/mask to match H,W
        orig = get_noop_transform()(image=img_np, mask=mask_np)
        orig_img = F.interpolate(
            orig["image"].float().unsqueeze(0) / 255.0,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        tmp_o_mask = orig["mask"].float()
        if tmp_o_mask.ndim == 2:
            tmp_o_mask = tmp_o_mask.unsqueeze(0)
        tmp_o_mask = tmp_o_mask.unsqueeze(0).repeat(1, 3, 1, 1) / 255.0
        orig_mask = F.interpolate(tmp_o_mask, size=(H, W), mode="nearest").squeeze(0)

        # --- 2) batch-level mixing simulation ---
        mix_img, mix_mask = aug_img.clone(), aug_mask.clone()
        batch_replay = {"transforms": []}

        # Mosaic
        if batch_cfg.mosaic.enable and random.random() < batch_cfg.mosaic.p and total >= 4:
            picks = [idx] + random.sample([i for i in range(total) if i != idx], 3)
            samples = [(aug_img, aug_mask)]
            for pi in picks:
                ip, mp = dataset.get_file_paths(pi) if hasattr(dataset, "get_file_paths") else (dataset.images[pi], dataset.masks[pi])
                oj = dataset.transform(
                    image=np.array(Image.open(ip).convert("RGB")),
                    mask=np.array(Image.open(mp).convert("L")),
                )
                ij = oj["image"].float() / 255.0
                mj = oj["mask"].float()
                if mj.ndim == 2:
                    mj = mj.unsqueeze(0)
                mj = mj / 255.0
                mj = mj.unsqueeze(0).repeat(1, 3, 1, 1).squeeze(0)
                samples.append((ij, mj))

            mos_imgs, mos_masks = mosaic_collate(samples, p=1.0)
            mix_img, mix_mask = mos_imgs[0], mos_masks[0]
            batch_replay["transforms"].append(
                {"__class_fullname__": "BatchMosaic", "applied": True}
            )

        # CutMix (manual, so we can draw the rectangle)
        if batch_cfg.cutmix.enable and random.random() < batch_cfg.cutmix.p:
            lam = np.random.beta(batch_cfg.cutmix.alpha, batch_cfg.cutmix.alpha)
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = np.random.randint(0, W)
            cy = np.random.randint(0, H)
            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            other = (idx + 1) % total
            ip, mp = dataset.get_file_paths(other) if hasattr(dataset, "get_file_paths") else (dataset.images[other], dataset.masks[other])
            o2 = dataset.transform(
                image=np.array(Image.open(ip).convert("RGB")),
                mask=np.array(Image.open(mp).convert("L")),
            )
            i2 = o2["image"].float() / 255.0
            m2 = o2["mask"].float()
            if m2.ndim == 2:
                m2 = m2.unsqueeze(0)
            m2 = m2 / 255.0
            m2 = m2.unsqueeze(0).repeat(1, 3, 1, 1).squeeze(0)

            mixed_img = mix_img.clone()
            mixed_mask = mix_mask.clone()
            mixed_img[:, y1:y2, x1:x2] = i2[:, y1:y2, x1:x2]
            mixed_mask[:, y1:y2, x1:x2] = m2[:, y1:y2, x1:x2]

            pil = to_pil_image(mixed_img)
            draw = ImageDraw.Draw(pil)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            mix_img = ToTensor()(pil)
            mix_mask = mixed_mask

            batch_replay["transforms"].append(
                {"__class_fullname__": "BatchCutMix", "applied": True}
            )

        # MixUp
        if batch_cfg.mixup.enable and random.random() < batch_cfg.mixup.p:
            other = (idx + 1) % total
            ip, mp = dataset.get_file_paths(other) if hasattr(dataset, "get_file_paths") else (dataset.images[other], dataset.masks[other])
            o2 = dataset.transform(
                image=np.array(Image.open(ip).convert("RGB")),
                mask=np.array(Image.open(mp).convert("L")),
            )
            i2 = o2["image"].float() / 255.0
            m2 = o2["mask"].float()
            if m2.ndim == 2:
                m2 = m2.unsqueeze(0)
            m2 = m2 / 255.0
            m2 = m2.unsqueeze(0).repeat(1, 3, 1, 1).squeeze(0)

            mix_imgs, mix_masks = mixup_collate([(mix_img, mix_mask), (i2, m2)], p=1.0, alpha=batch_cfg.mixup.alpha)
            mix_img, mix_mask = mix_imgs[0], mix_masks[0]
            batch_replay["transforms"].append(
                {"__class_fullname__": "BatchMixUp", "applied": True}
            )

        # --- 3) render legends (always include batch legend) ---
        img_leg = render_legend(
            replay1, height=H, full_width=W,
            mask_mode=False, mask_names=set(),
            font_divisor=font_div, min_font_size=min_fs,
        )
        mask_leg = render_legend(
            replay1, height=H, full_width=W,
            mask_mode=True, mask_names=SPATIAL_MASK_SAFE,
            font_divisor=font_div, min_font_size=min_fs,
        )
        batch_leg = render_legend(
            batch_replay, height=H, full_width=W,
            mask_mode=False, mask_names=set(),
            font_divisor=font_div, min_font_size=min_fs,
        )

        # --- 4) assemble rows with exactly 5 columns each ---
        row1 = [orig_img, aug_img, img_leg, mix_img, batch_leg]
        row2 = [orig_mask, aug_mask, mask_leg, mix_mask, batch_leg]

        cells = row1 + row2 if 'cells' not in locals() else cells + row1 + row2

    # build & save grid with fixed 5 columns
    ncols = 5
    grid = make_grid(cells, nrow=ncols, padding=4)
    out_dir = Path(HydraConfig.get().runtime.output_dir) / "image_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "aug_visualization.png"
    save_image(grid, str(out_file))
    print(f"Saved comparison grid to: {out_file}")

    # Warn once if wrappers are present (now common with grouped config)
    print(
        "Note: Group wrappers (SomeOf/OneOf/Sequential) are shown as blocks in the legend. "
        "Only the actually applied inner transforms are recorded by Albumentations."
    )

    # log to all configured loggers
    for lcfg in logger_cfgs:
        logger = hydra.utils.instantiate(lcfg)
        exp = getattr(logger, "experiment", None)
        if exp and hasattr(exp, "log"):
            exp.log({"train/aug_visualization": [wandb.Image(str(out_file))]})


def run_viz_augments(cfg: DictConfig) -> None:
    """
    Standalone augmentation preview:
    builds a minimal train DataLoader and renders the augmentation grid.
    """
    ds = FieldDataset(cfg, mode="train")
    collate = get_batch_collate_fn(cfg.augment.train.batch)
    loader = DataLoader(
        ds,
        batch_size=max(1, min(8, len(ds))),
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=collate,
    )

    vis_augmentation_batch(
        loader,
        cfg.train.logger,
        num_samples=cfg.augment.augmentation_visualizer.num_samples,
    )
