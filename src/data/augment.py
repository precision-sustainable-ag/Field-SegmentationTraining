# src/data/augment.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any


def _build_group(
    cfg_group: Dict[str, Any],
    class_map: Dict[str, Any],
    extra: Dict[str, Any] = None
) -> list:
    """
    Helper: instantiate all enabled transforms in a group.

    Args:
        cfg_group:   The config subtree for this group (spatial or pixel).
        class_map:   Mapping from config keys to Albumentations classes.
        extra:       Optional dict of extra params for grouped transforms.

    Returns:
        A list of instantiated Albumentations transform objects.
    """
    ops = []
    for key, cls in class_map.items():
        spec = cfg_group.get(key, {})
        if not spec.get("enable", False):
            continue
        # Collect parameters except the 'enable' flag
        params = {k: v for k, v in spec.items() if k != "enable"}
        # Merge in any extras (e.g. weightings for OneOf, etc.)
        if extra and key in extra:
            params.update(extra[key])
        ops.append(cls(**params))
    return ops


def get_train_transforms(cfg):
    """
    Build the full training augmentation pipeline:
      1) Spatial-level ops applied to both image & mask
      2) Pixel-level ops applied to image only
      3) Conversion to tensor
      4) Replay info for introspection

    Relies on `cfg.augment.train` for enabled flags and parameters.
    """
    t = cfg.augment.train

    # ─── Spatial transforms (image + mask) ───────────────────────────────
    spat_map = {
        "random_crop":        A.RandomCrop,
        "horizontal_flip":    A.HorizontalFlip,
        "vertical_flip":      A.VerticalFlip,
        "random_rotate90":    A.RandomRotate90,
        "affine":             A.Affine,
        "elastic_transform":  A.ElasticTransform,
        "grid_distortion":    A.GridDistortion,
        "perspective":        A.Perspective,
        "optical_distortion": A.OpticalDistortion,
        "random_scale":       A.RandomScale,
        "shift_scale_rotate": A.ShiftScaleRotate,
    }
    spatial_ops = _build_group(t.spatial, spat_map)

    # # ─── SomeOf for spatial ops ───────────────────────────────────────
    # so = getattr(t.spatial, "some_of", None)
    # if so and so.enable:
    #     # Build the list of Albumentations transform instances
    #     some_list = []
    #     for key in so.ops:
    #         spec   = t.spatial.get(key, {})
    #         params = {k:v for k,v in spec.items() if k not in ("enable",)}
    #         some_list.append(spat_map[key](**params))
    #     spatial_ops.append(
    #         A.SomeOf(
    #             some_list,
    #             n = int(so.n),
    #             replace = False,
    #             p = float(so.p)
    #         )
    #     )

    # ─── Pixel-level transforms (image only) ─────────────────────────────
    pix_map = {
        "color_jitter":             A.ColorJitter,
        "random_brightness_contrast":A.RandomBrightnessContrast,
        "random_gamma":             A.RandomGamma,
        "gauss_noise":              A.GaussNoise,
        "multiplicative_noise":     A.MultiplicativeNoise,
        "iso_noise":                A.ISONoise,
        "clahe":                    A.CLAHE,
        "image_compression":        A.ImageCompression,
        "rgb_shift":                A.RGBShift,
        "channel_shuffle":          A.ChannelShuffle,
        "coarse_dropout":           A.CoarseDropout,
    }
    pixel_ops = _build_group(t.pixel, pix_map)

    # # ─── SomeOf for pixel ops ─────────────────────────────────────────
    # po = getattr(t.pixel, "some_of", None)
    # if po and po.enable:
    #     some_list = []
    #     for key in po.ops:
    #         spec   = t.pixel.get(key, {})
    #         params = {k:v for k,v in spec.items() if k not in ("enable",)}
    #         some_list.append(pix_map[key](**params))
    #     pixel_ops.append(
    #         A.SomeOf(
    #             some_list,
    #             n = int(po.n),
    #             replace = False,
    #             p = float(po.p)
    #         )
    #     )

    # ─── Assemble final pipeline ─────────────────────────────────────────
    # - replay=True captures which transforms actually ran & their params
    # - additional_targets ensures masks go through only spatial ops
    # 1) spatial_ops + pixel_ops
    # 2) resize *always* to target size so DataLoader can batch
    # 3) to-tensor + replay capture
    H = int(cfg.augment.train.img_size.height)
    W = int(cfg.augment.train.img_size.width)

    pipeline = spatial_ops + pixel_ops + [
        # ensure fixed output dimensions
        A.Resize(height=H, width=W, p=1.0),
        ToTensorV2()
    ]

    return A.ReplayCompose(
        transforms=pipeline,
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(cfg) -> A.Compose:
    """
    Validation transforms: pad/resize only, then to tensor.
    """
    t = cfg.augment.val
    ops = []
    if t.enable:
        height, width = t.img_size.height, t.img_size.width
        ops.append(A.PadIfNeeded(min_height=height, min_width=width, p=1.0))
        ops.append(ToTensorV2())
    return A.Compose(ops)


def get_test_transforms(cfg) -> A.Compose:
    """
    Test transforms: same as validation by default.
    """
    return get_val_transforms(cfg)


def get_noop_transform() -> A.Compose:
    """
    No-op pipeline: returns image & mask untouched (but as tensors).
    """
    return A.Compose(
        [A.NoOp(), ToTensorV2()],
        additional_targets={"mask": "mask"}
    )
