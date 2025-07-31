# src/data/augment.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any

def _build_group(cfg_group: Dict[str, Any], class_map: Dict[str, Any], extra: Dict[str, Any]=None):
    """Helper: pick enabled transforms from a config group."""
    ts = []
    for key, cls in class_map.items():
        spec = cfg_group.get(key, {})
        if spec.get("enable", False):
            params = {k:v for k,v in spec.items() if k!="enable"}
            if extra and key in extra:
                params.update(extra[key])
            ts.append(cls(**params))
    return ts

def get_train_transforms(cfg):
    t = cfg.augment.train

    # ─── spatial transforms (both image+mask) ────────────────────────────────
    spatial_ops = []
    spat_map = {
      "horizontal_flip":  A.HorizontalFlip,
      "vertical_flip":    A.VerticalFlip,
      "random_rotate90":  A.RandomRotate90,
      "random_crop":      A.RandomCrop,
      "affine":           A.Affine,
      "elastic_transform":A.ElasticTransform,
      "grid_distortion":  A.GridDistortion,
      "perspective":      A.Perspective,
      "optical_distortion":A.OpticalDistortion,
    }
    for key, cls in spat_map.items():
        c = t.spatial.get(key, {})
        if c.get("enable", False):
            params = {k:v for k,v in c.items() if k!="enable"}
            spatial_ops.append(cls(**params))

    # ─── pixel‐level transforms (image only) ─────────────────────────────────
    pixel_ops = []
    pix_map = {
      "color_jitter":          A.ColorJitter,
      "random_brightness_contrast":A.RandomBrightnessContrast,
      "random_gamma":           A.RandomGamma,
      "clahe":                  A.CLAHE,
      "gauss_noise":            A.GaussNoise,
      "multiplicative_noise":   A.MultiplicativeNoise,
      "iso_noise":              A.ISONoise,
      "image_compression":      A.ImageCompression,
      "rgb_shift":              A.RGBShift,
      "channel_shuffle":        A.ChannelShuffle,
    }
    for key, cls in pix_map.items():
        c = t.pixel.get(key, {})
        if c.get("enable", False):
            params = {k:v for k,v in c.items() if k!="enable"}
            pixel_ops.append(cls(**params))

    # ─── build final Compose ────────────────────────────────────────────────
    # spatial + pixel, then ToTensor, masks carried via additional_targets
    return A.Compose(
      spatial_ops + pixel_ops + [ToTensorV2()],
      additional_targets={"mask": "mask"},
    )


def get_val_transforms(cfg: Any) -> A.Compose:
    t = cfg.augment.val
    ts = []
    # we’ll just resize/pad to target size
    if t.enable:
        size = (t.img_size.height, t.img_size.width)
        ts.append(A.PadIfNeeded(min_height=size[0], min_width=size[1], p=1.0))
        ts.append(ToTensorV2())
    return A.Compose(ts)

def get_test_transforms(cfg: Any) -> A.Compose:
    # same as val by default
    return get_val_transforms(cfg)

def get_noop_transform() -> A.Compose:
    """
    Returns an Albumentations transform that performs no augmentation.
    Useful as a fallback when augmentations are disabled.
    """
    return A.Compose([
        A.NoOp(),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})
