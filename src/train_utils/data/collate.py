# src/train_utils/data/collate.py

"""
Batch‐level augmentation collate functions for segmentation.

Provides MixUp, CutMix, and Mosaic variants that operate on
torch.Tensor batches of (image, mask) pairs. Each transform
is designed to modify *exactly one* sample per batch
(if applied), leaving the rest untouched, to preserve batch
diversity.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from typing import List, Tuple, Callable, Any

def mixup_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    p: float,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MixUp augmentation over the entire batch with probability p.

    For MixUp, we sample λ ~ Beta(alpha, alpha), then form
    mixed = λ * x + (1−λ) * x_shuffled for both images and masks,
    broadcasting across the batch. If the random draw fails,
    returns the original batch untouched.

    Args:
        batch: List of (image, mask) tuples.
               - image: Tensor[C_img, H, W]
               - mask:  Tensor[C_mask, H, W]
        p:      Probability of applying MixUp.
        alpha:  Hyperparameter for the Beta distribution.

    Returns:
        imgs:  Tensor[B, C_img, H, W]
        masks: Tensor[B, C_mask, H, W]
    """
    # Unzip and stack to [B, C, H, W]
    imgs, masks = zip(*batch)
    imgs  = torch.stack(imgs,  0)
    masks = torch.stack(masks, 0)

    # Apply MixUp with probability p
    if random.random() < p:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(imgs.size(0))  # shuffle indices
        mixed_imgs  = lam * imgs + (1 - lam) * imgs[idx]
        mixed_masks = lam * masks + (1 - lam) * masks[idx]
        return mixed_imgs, mixed_masks

    # Fallback: return original batch
    return imgs, masks


def cutmix_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    p: float,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single‐sample CutMix: replace one sample in the batch.

    With probability p, selects one `out_idx` to modify and
    one `other_idx` to pull a random rectangular patch from.
    The box size is determined by λ ~ Beta(alpha, alpha)
    (area ratio), and the patch is copied into both image
    and mask channels. Other batch entries remain identical.

    Args:
        batch: List of (image, mask) tuples.
        p:      Probability of performing CutMix.
        alpha:  Beta distribution parameter for patch size.

    Returns:
        imgs:  Tensor[B, C_img, H, W]
        masks: Tensor[B, C_mask, H, W]
    """
    # Stack to tensors of shape [B, C, H, W]
    imgs, masks = zip(*batch)
    imgs  = torch.stack(imgs,  0)
    masks = torch.stack(masks, 0)
    B, C_img, H, W = imgs.shape
    _, C_mask, _, _ = masks.shape

    # Only apply if batch has at least two samples
    if B >= 2:
        # 1) Select which sample to replace, and which to source from
        out_idx   = random.randrange(B)
        other_idx = random.choice([i for i in range(B) if i != out_idx])

        # 2) Sample lambda and compute cut box coordinates
        lam = np.random.beta(alpha, alpha)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w   = int(W * cut_rat)
        cut_h   = int(H * cut_rat)
        cx = random.randrange(W)
        cy = random.randrange(H)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        # 3) Clone the target sample and swap in the patch
        new_img  = imgs[out_idx].clone()
        new_mask = masks[out_idx].clone()
        new_img[:,  y1:y2, x1:x2] = imgs[other_idx][:,  y1:y2, x1:x2]
        new_mask[:, y1:y2, x1:x2] = masks[other_idx][:, y1:y2, x1:x2]

        # 4) Write back into the batch tensor
        imgs[out_idx]  = new_img
        masks[out_idx] = new_mask

    return imgs, masks


def mosaic_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    p: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single‐sample Mosaic: replace one sample with a 2×2 tile mosaic.

    With probability p, picks four *distinct* source samples,
    constructs a 2H×2W canvas, tiles them in order, downsamples
    back to (H, W), and swaps that mosaic into a random batch slot.
    All other samples remain untouched.

    Args:
        batch: List of (image, mask) tuples.
        p:      Probability of performing Mosaic.

    Returns:
        imgs:  Tensor[B, C_img, H, W]
        masks: Tensor[B, C_mask, H, W]
    """
    # 1) stack to [B, C, H, W]
    imgs, masks = zip(*batch)
    imgs  = torch.stack(imgs,  0)
    masks = torch.stack(masks, 0)
    B, C_img, H, W = imgs.shape
    _, C_mask, _, _ = masks.shape

    # 2) apply with probability p and if enough samples
    if random.random() < p and B >= 4:
        # a) pick four different samples
        src_idxs = random.sample(range(B), 4)
        q_imgs   = imgs[src_idxs]   # [4, C_img, H, W]
        q_masks  = masks[src_idxs]  # [4, C_mask, H, W]

        # b) build 2H x 2W canvases
        canvas_img  = torch.zeros((C_img, 2*H, 2*W),  device=imgs.device, dtype=imgs.dtype)
        canvas_mask = torch.zeros((C_mask, 2*H, 2*W), device=masks.device, dtype=masks.dtype)

        # c) paste the four quadrants in fixed order
        canvas_img[:,         :H,          :W]    = q_imgs[0]
        canvas_img[:,         :H,     W:2*W]      = q_imgs[1]
        canvas_img[:,    H:2*H,          :W]      = q_imgs[2]
        canvas_img[:,    H:2*H,     W:2*W]        = q_imgs[3]

        canvas_mask[:,        :H,          :W]    = q_masks[0]
        canvas_mask[:,        :H,     W:2*W]       = q_masks[1]
        canvas_mask[:,   H:2*H,          :W]       = q_masks[2]
        canvas_mask[:,   H:2*H,     W:2*W]         = q_masks[3]

        # d) downsample back to original resolution
        mos_img  = F.interpolate(
            canvas_img .unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        mos_mask = F.interpolate(
            canvas_mask.unsqueeze(0),
            size=(H, W),
            mode="nearest"
        ).squeeze(0)

        # e) replace exactly one random batch slot
        out_idx = random.randrange(B)
        imgs[out_idx]  = mos_img
        masks[out_idx] = mos_mask

    return imgs, masks


def get_batch_collate_fn(
    batch_cfg: Any
) -> Callable[[List[Tuple[torch.Tensor, torch.Tensor]]], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a collate_fn for DataLoader applying MixUp, CutMix, Mosaic.

    The returned function will:
      1. default_collate the incoming list
      2. run mosaic_collate on one sample if enabled
      3. run cutmix_collate on one sample if enabled
      4. run mixup_collate over whole batch if enabled

    Args:
        batch_cfg: Config object with
            .mosaic.enable/.p
            .cutmix.enable/.p/.alpha
            .mixup.enable/.p/.alpha

    Returns:
        collate_fn compatible with torch.utils.data.DataLoader
    """
    def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        # Step 1: default stacking
        imgs, masks = default_collate(batch)

        # Step 2: single-sample mosaic
        if batch_cfg.mosaic.enable:
            imgs, masks = mosaic_collate(
                list(zip(imgs, masks)),
                p=batch_cfg.mosaic.p
            )

        # Step 3: single-sample cutmix
        if batch_cfg.cutmix.enable:
            imgs, masks = cutmix_collate(
                list(zip(imgs, masks)),
                p=batch_cfg.cutmix.p,
                alpha=batch_cfg.cutmix.alpha
            )

        # Step 4: batch-wide mixup
        if batch_cfg.mixup.enable:
            imgs, masks = mixup_collate(
                list(zip(imgs, masks)),
                p=batch_cfg.mixup.p,
                alpha=batch_cfg.mixup.alpha
            )

        return imgs, masks

    # If batch-level mixing is fully disabled, return the plain collate
    return default_collate if not batch_cfg.enable else collate
