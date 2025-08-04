# src/data/collate.py

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

def mixup_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    p: float,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate + MixUp. Returns (imgs, masks) always.
    """
    imgs, masks = zip(*batch)
    imgs  = torch.stack(imgs, 0)
    masks = torch.stack(masks, 0)

    if random.random() < p:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(imgs.size(0))
        mixed_imgs  = lam * imgs + (1 - lam) * imgs[idx]
        mixed_masks = lam * masks + (1 - lam) * masks[idx]
        return mixed_imgs, mixed_masks

    return imgs, masks


def cutmix_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    p: float,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate + CutMix. Always returns (imgs, masks).
    """
    imgs, masks = zip(*batch)
    imgs  = torch.stack(imgs, 0)
    masks = torch.stack(masks, 0)

    if random.random() < p:
        lam = np.random.beta(alpha, alpha)
        B, C, H, W = imgs.shape
        idx = torch.randperm(B)

        # determine random box
        cut_rat = np.sqrt(1. - lam)
        cut_w   = int(W * cut_rat)
        cut_h   = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # swap the patch in both image and mask
        imgs[:, :, y1:y2, x1:x2]  = imgs[idx, :, y1:y2, x1:x2]
        masks[:, :, y1:y2, x1:x2] = masks[idx, :, y1:y2, x1:x2]

        # return the in-place mixed result
        return imgs, masks

    # fallback: no CutMix
    return imgs, masks


def mosaic_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    p: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate + Mosaic of exactly 4 samples into one 2×2 tile, then replicate
    back to original batch size. Always returns (imgs, masks).
    """
    if random.random() > p or len(batch) < 4:
        return default_collate(batch)

    # pick exactly 4 random samples
    samples = random.sample(batch, 4)
    imgs, masks = zip(*samples)
    
    # image channels & dims
    C_img, H, W = imgs[0].shape
    # mask may have 1 or more channels
    C_mask = masks[0].shape[0]

    # build mosaic canvas
    canvas_img  = torch.zeros((C_img, 2*H, 2*W), dtype=imgs[0].dtype)
    canvas_mask = torch.zeros((C_mask, 2*H, 2*W), dtype=masks[0].dtype)

    # place 4 quadrants
    canvas_img[:, :H, :W]   = imgs[0]
    canvas_img[:, :H, W: ]  = imgs[1]
    canvas_img[:, H: , :W]  = imgs[2]
    canvas_img[:, H: , W: ]  = imgs[3]

    canvas_mask[:, :H, :W]   = masks[0]
    canvas_mask[:, :H, W: ]  = masks[1]
    canvas_mask[:, H: , :W]  = masks[2]
    canvas_mask[:, H: , W: ] = masks[3]

    # replicate mosaic to match original batch size
    B = len(batch)
    out_imgs  = canvas_img .unsqueeze(0).repeat(B, 1, 1, 1)
    out_masks = canvas_mask.unsqueeze(0).repeat(B, 1, 1, 1)

    # ---- HERE: resize back down to (H,W) ----
    out_imgs  = F.interpolate(out_imgs,  size=(H, W), mode="bilinear",   align_corners=False)
    out_masks = F.interpolate(out_masks, size=(H, W), mode="nearest")

    return out_imgs, out_masks


def get_batch_collate_fn(
    batch_cfg: Any
) -> Callable[[List[Tuple[torch.Tensor, torch.Tensor]]], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Return a collate_fn that applies CutMix and Mosaic according to batch_cfg.
    MixUp is left unchanged but may be disabled in your config.
    """
    def collate(batch):
        # default stacking
        imgs, masks = default_collate(batch)

        # Mosaic
        if batch_cfg.mosaic.enable:
            imgs, masks = mosaic_collate(
                list(zip(imgs, masks)),
                p=batch_cfg.mosaic.p
            )

        # CutMix
        if batch_cfg.cutmix.enable:
            imgs, masks = cutmix_collate(
                list(zip(imgs, masks)),
                p=batch_cfg.cutmix.p,
                alpha=batch_cfg.cutmix.alpha
            )

        # MixUp
        if batch_cfg.mixup.enable:
            imgs, masks = mixup_collate(
                list(zip(imgs, masks)),
                p=batch_cfg.mixup.p,
                alpha=batch_cfg.mixup.alpha
            )

        return imgs, masks

    # If no batch mixing is enabled, return the default collate function
    return default_collate if not batch_cfg.enable else collate
