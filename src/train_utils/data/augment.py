# src/train_utils/data/augment.py

from typing import Dict, Any, List, Optional, Callable, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ───────────────────────────────
# Utilities
# ───────────────────────────────

def _pop(d: Dict[str, Any], key: str, default=None):
    """Pop key if it exists; otherwise return default (without raising)."""
    if not isinstance(d, dict):
        return default
    return d.pop(key, default)


def _translate_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # v2-native config: just return a shallow copy
    # add any changes to go from v1 to v2 if necessary
    return dict(params) if params else {}


def _maybe(name: str) -> Optional[Any]:
    """Return Albumentations transform class by name if available; else None."""
    return getattr(A, name, None)


def _wrap_mode(ops: List[A.BasicTransform], mode: Dict[str, Any]) -> Optional[A.BasicTransform]:
    """
    Wrap a list of transforms according to subgroup `mode`.
    - type: one_of | some_of | all
    - p: probability of applying the whole block
    - n, replace: only for some_of
    """
    if not ops:
        return None
    mtype = (mode.get("type") or "all").lower()
    p = float(mode.get("p", 1.0))
    if mtype == "one_of":
        return A.OneOf(ops, p=p)
    elif mtype == "some_of":
        n = int(mode.get("n", max(1, len(ops) // 2)))
        replace = bool(mode.get("replace", False))
        return A.SomeOf(ops, n=n, replace=replace, p=p)
    else:  # "all"
        # Apply all, but honor the group's probability p
        return A.Sequential(ops, p=p)

def _cv2_border_code(mode: Any) -> Any:
    """
    Map human-friendly strings to cv2 border codes.
    Accepts either string or int; returns input unchanged if cv2 unavailable.
    """
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is None or mode is None:
        return mode

    if isinstance(mode, int):
        return mode

    table = {
        "constant":   getattr(cv2, "BORDER_CONSTANT", 0),
        "reflect":    getattr(cv2, "BORDER_REFLECT", 2),
        "reflect101": getattr(cv2, "BORDER_REFLECT_101", 4),
        "replicate":  getattr(cv2, "BORDER_REPLICATE", 1),
        "wrap":       getattr(cv2, "BORDER_WRAP", 3),
    }
    key = str(mode).strip().lower()
    return table.get(key, getattr(cv2, "BORDER_CONSTANT", 0))

def _cv2_interp_code(name: any) -> any:
    """
    Map user-friendly strings to cv2 interpolation codes.
    Accepts string or int; returns input unchanged if cv2 unavailable or already int.
    """
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is None or name is None or isinstance(name, int):
        return name

    lut = {
        "nearest":  getattr(cv2, "INTER_NEAREST", 0),
        "linear":   getattr(cv2, "INTER_LINEAR", 1),
        "cubic":    getattr(cv2, "INTER_CUBIC", 2),
        "area":     getattr(cv2, "INTER_AREA", 3),
        "lanczos4": getattr(cv2, "INTER_LANCZOS4", 4),
    }
    return lut.get(str(name).strip().lower(), getattr(cv2, "INTER_LINEAR", 1))


def _normalize_area_for_downscale(val: any) -> any:
    """
    Ensure area_for_downscale is one of: None, "image", "image_mask".
    Accepts strings like 'none', 'image', 'image_mask'.
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "none", "null"):
        return None
    if s in ("image", "image_mask"):
        return s
    # default: keep None rather than invalid string
    return None
    

# ───────────────────────────────
# Builders for specific sub-groups
# ───────────────────────────────

def _build_initial_resize_crop(cfg: Dict[str, Any], H: int, W: int) -> List[A.BasicTransform]:
    """
    Build candidate strategies for the 'initial_resize_crop' subgroup.
    Each candidate is a single transform or an A.Sequential of multiple.
    """
    candidates: List[A.BasicTransform] = []

    # ---- RandomCrop (full param coverage) ----
    spec = cfg.get("random_crop", {})
    if spec.get("enable", False):
        # collect params, apply defaults, and convert border_mode if present
        params = {k: v for k, v in spec.items() if k not in ("enable")}
        height = int(params.pop("height", H))
        width  = int(params.pop("width",  W))

        # Handle optional padding behavior
        pad_if_needed = bool(params.pop("pad_if_needed", False))
        border_mode   = _cv2_border_code(params.pop("border_mode", None))
        # Albumentations v2 RandomCrop supports: pad_if_needed, border_mode, fill, fill_mask, pad_position, p

        rc = A.RandomCrop(
            height=height,
            width=width,
            pad_if_needed=pad_if_needed,
            border_mode=border_mode,
            **params  # includes: fill, fill_mask, pad_position, p (and any future-safe keys)
        )
        candidates.append(rc)

    # ---- SmallestMaxSize -> RandomCrop ----
    spec = cfg.get("smallest_max_size_then_random_crop", {})
    if spec.get("enable", False):
        pval = float(spec.get("p", 1.0))

        # decide which size constraint to use
        max_size = spec.get("max_size", None)
        max_size_hw = spec.get("max_size_hw", None)  # list/tuple or None

        # prepare interpolation args
        interp = _cv2_interp_code(spec.get("interpolation", "linear"))
        mask_interp = _cv2_interp_code(spec.get("mask_interpolation", "nearest"))
        area = _normalize_area_for_downscale(spec.get("area_for_downscale", None))

        # final crop size
        final_h = int(spec.get("final_height", H))
        final_w = int(spec.get("final_width",  W))

        smm_kwargs = {
            "interpolation": interp,
            "mask_interpolation": mask_interp,
            "area_for_downscale": area,
            "p": 1.0,
        }
        if max_size_hw is not None:
            # accept list/tuple like [H, W] or [H, null]
            if isinstance(max_size_hw, (list, tuple)) and len(max_size_hw) == 2:
                h0 = None if max_size_hw[0] in (None, "null") else int(max_size_hw[0])
                w0 = None if max_size_hw[1] in (None, "null") else int(max_size_hw[1])
                smm_kwargs["max_size_hw"] = (h0, w0)
            else:
                # if malformed, fall back to max_size
                if max_size is None:
                    max_size = min(H, W)
        if max_size is not None and "max_size_hw" not in smm_kwargs:
            # allow int or list of ints
            smm_kwargs["max_size"] = max_size

        seq = A.Sequential(
            [
                A.SmallestMaxSize(**smm_kwargs),
                A.RandomCrop(height=final_h, width=final_w, p=1.0),
            ],
            p=pval,
        )
        candidates.append(seq)

    # ---- SmallestMaxSize only (standalone) ----
    spec = cfg.get("smallest_max_size", {})
    if spec.get("enable", False):
        pval = float(spec.get("p", 1.0))
        max_size = spec.get("max_size", None)
        max_size_hw = spec.get("max_size_hw", None)
        interp = _cv2_interp_code(spec.get("interpolation", "linear"))
        mask_interp = _cv2_interp_code(spec.get("mask_interpolation", "nearest"))
        area = _normalize_area_for_downscale(spec.get("area_for_downscale", None))

        smm_kwargs = {
            "interpolation": interp,
            "mask_interpolation": mask_interp,
            "area_for_downscale": area,
            "p": pval,
        }
        if max_size_hw is not None:
            if isinstance(max_size_hw, (list, tuple)) and len(max_size_hw) == 2:
                h0 = None if max_size_hw[0] in (None, "null") else int(max_size_hw[0])
                w0 = None if max_size_hw[1] in (None, "null") else int(max_size_hw[1])
                smm_kwargs["max_size_hw"] = (h0, w0)
        elif max_size is not None:
            smm_kwargs["max_size"] = max_size
        else:
            smm_kwargs["max_size"] = min(H, W)  # sensible default

        candidates.append(A.SmallestMaxSize(**smm_kwargs))

    # ---- LongestMaxSize -> PadIfNeeded ----
    spec = cfg.get("longest_max_size_then_pad_if_needed", {})
    if spec.get("enable", False):
        pval = float(spec.get("p", 1.0))

        # Which constraint to use
        max_size = spec.get("max_size", None)
        max_size_hw = spec.get("max_size_hw", None)

        # Interpolation settings
        interp       = _cv2_interp_code(spec.get("interpolation", "linear"))
        mask_interp  = _cv2_interp_code(spec.get("mask_interpolation", "nearest"))
        area         = _normalize_area_for_downscale(spec.get("area_for_downscale", None))

        # Padding target
        pad_h = int(spec.get("pad_min_height", H))
        pad_w = int(spec.get("pad_min_width",  W))
        pad_border = _cv2_border_code(spec.get("pad_border_mode", "constant"))
        # (Optional) If you later want fill values/position:
        # pad_fill      = spec.get("pad_fill", None)
        # pad_fill_mask = spec.get("pad_fill_mask", None)
        # pad_position  = spec.get("pad_position", None)

        lms_kwargs = {
            "interpolation": interp,
            "mask_interpolation": mask_interp,
            "area_for_downscale": area,
            "p": 1.0,
        }
        if max_size_hw is not None:
            if isinstance(max_size_hw, (list, tuple)) and len(max_size_hw) == 2:
                h0 = None if max_size_hw[0] in (None, "null") else int(max_size_hw[0])
                w0 = None if max_size_hw[1] in (None, "null") else int(max_size_hw[1])
                lms_kwargs["max_size_hw"] = (h0, w0)
        elif max_size is not None:
            lms_kwargs["max_size"] = max_size
        else:
            lms_kwargs["max_size"] = max(H, W)  # sensible default for "longest"

        # Build sequence: resize to longest, then pad to (pad_h, pad_w)
        pad_kwargs = {
            "min_height": pad_h,
            "min_width":  pad_w,
            "border_mode": pad_border,
            "p": 1.0,
        }
        # If you later want to support pad fill/position safely, uncomment and map:
        # if pad_fill is not None:
        #     pad_kwargs["fill"] = pad_fill
        # if pad_fill_mask is not None:
        #     pad_kwargs["fill_mask"] = pad_fill_mask
        # if pad_position is not None:
        #     pad_kwargs["position"] = pad_position

        seq = A.Sequential(
            [
                A.LongestMaxSize(**lms_kwargs),
                A.PadIfNeeded(**pad_kwargs),
            ],
            p=pval,
        )
        candidates.append(seq)

    # ---- LongestMaxSize only ----
    spec = cfg.get("longest_max_size", {})
    if spec.get("enable", False):
        pval = float(spec.get("p", 1.0))
        max_size = spec.get("max_size", None)
        max_size_hw = spec.get("max_size_hw", None)

        interp      = _cv2_interp_code(spec.get("interpolation", "linear"))
        mask_interp = _cv2_interp_code(spec.get("mask_interpolation", "nearest"))
        area        = _normalize_area_for_downscale(spec.get("area_for_downscale", None))

        lms_kwargs = {
            "interpolation": interp,
            "mask_interpolation": mask_interp,
            "area_for_downscale": area,
            "p": pval,
        }
        if max_size_hw is not None:
            if isinstance(max_size_hw, (list, tuple)) and len(max_size_hw) == 2:
                h0 = None if max_size_hw[0] in (None, "null") else int(max_size_hw[0])
                w0 = None if max_size_hw[1] in (None, "null") else int(max_size_hw[1])
                lms_kwargs["max_size_hw"] = (h0, w0)
        elif max_size is not None:
            lms_kwargs["max_size"] = max_size
        else:
            lms_kwargs["max_size"] = max(H, W)

        candidates.append(A.LongestMaxSize(**lms_kwargs))

    return candidates


def _build_basic_geometric(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    ops: List[A.BasicTransform] = []

    for name, cls_name in [
        ("horizontal_flip", "HorizontalFlip"),
        ("vertical_flip", "VerticalFlip"),
        ("random_rotate90", "RandomRotate90"),
    ]:
        spec = cfg.get(name, {})
        if spec.get("enable", False):
            cls = _maybe(cls_name)
            if cls:
                params = _translate_params(name, {k: v for k, v in spec.items() if k != "enable"})
                ops.append(cls(**params))
    return ops


def _build_affine_perspective(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    entries = [
        ("affine",             "Affine"),
        ("perspective",        "Perspective"),
        ("shift_scale_rotate", "ShiftScaleRotate"),
        ("optical_distortion", "OpticalDistortion"),
        ("random_scale",       "RandomScale"),
        # non‑linear warps placed at the end of the group
        ("grid_distortion",    "GridDistortion"),
        ("elastic_transform",  "ElasticTransform"),
        ("thin_plate_spline",  "ThinPlateSpline"),  # may not exist in some installs
    ]
    ops: List[A.BasicTransform] = []
    for key, cls_name in entries:
        spec = cfg.get(key, {})
        if spec.get("enable", False):
            cls = _maybe(cls_name)
            if cls:
                params = _translate_params(key, {k: v for k, v in spec.items() if k != "enable"})
                ops.append(cls(**params))
    return ops


def _build_dropout_occlusion(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    ops: List[A.BasicTransform] = []

    # CoarseDropout (dual: apply also to masks)
    spec = cfg.get("coarse_dropout", {})
    if spec.get("enable", False):
        params = _translate_params("coarse_dropout", {k: v for k, v in spec.items() if k != "enable"})
        ops.append(CoarseDropoutDual(**params))

    # GridDropout
    spec = cfg.get("grid_dropout", {})
    if spec.get("enable", False):
        cls = _maybe("GridDropout")
        if cls:
            params = _translate_params("grid_dropout", {k: v for k, v in spec.items() if k != "enable"})
            ops.append(cls(**params))

    # RandomErasing (if available in albumentations). If not, approximate with CoarseDropout.
    spec = cfg.get("random_erasing", {})
    if spec.get("enable", False):
        cls = _maybe("RandomErasing")
        if cls:
            params = _translate_params("random_erasing", {k: v for k, v in spec.items() if k != "enable"})
            ops.append(cls(**params))
        else:
            # Fallback approximation using CoarseDropoutDual with a single hole sized by scale/ratio
            # (Albumentations RandomErasing is not always available)
            scale = spec.get("scale", [0.02, 0.10])
            ratio = spec.get("ratio", [0.3, 3.3])
            approx = CoarseDropoutDual(max_holes=1, p=float(spec.get("p", 0.5)))
            ops.append(approx)

    return ops


def _build_color_space_reduction(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    ops: List[A.BasicTransform] = []
    # ToGray
    spec = cfg.get("to_gray", {})
    if spec.get("enable", False):
        cls = _maybe("ToGray")
        if cls:
            params = _translate_params("to_gray", {k: v for k, v in spec.items() if k != "enable"})
            ops.append(cls(**params))
    # ChannelDropout
    spec = cfg.get("channel_dropout", {})
    if spec.get("enable", False):
        cls = _maybe("ChannelDropout")
        if cls:
            params = _translate_params("channel_dropout", {k: v for k, v in spec.items() if k != "enable"})
            ops.append(cls(**params))
    return ops


def _build_color_augmentations(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    mapping = {
        "random_brightness_contrast": "RandomBrightnessContrast",
        "color_jitter":               "ColorJitter",
        "hue_saturation_value":       "HueSaturationValue",
        "random_gamma":               "RandomGamma",
        "rgb_shift":                  "RGBShift",
        "channel_shuffle":            "ChannelShuffle",
        "planckian_jitter":           "PlanckianJitter",   # NEW (if available)
    }
    ops: List[A.BasicTransform] = []
    for key, cls_name in mapping.items():
        spec = cfg.get(key, {})
        if spec.get("enable", False):
            cls = _maybe(cls_name)
            if cls:
                params = _translate_params(key, {k: v for k, v in spec.items() if k != "enable"})
                ops.append(cls(**params))
    return ops



def _build_blur(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    mapping = {
        "gaussian_blur": "GaussianBlur",
        "median_blur":   "MedianBlur",
        "motion_blur":   "MotionBlur",
        "advanced_blur": "AdvancedBlur",  # NEW (if available)
        "zoom_blur":     "ZoomBlur",      # NEW (if available)
    }
    ops: List[A.BasicTransform] = []
    for key, cls_name in mapping.items():
        spec = cfg.get(key, {})
        if spec.get("enable", False):
            cls = _maybe(cls_name)
            if cls:
                params = _translate_params(key, {k: v for k, v in spec.items() if k != "enable"})
                ops.append(cls(**params))
    return ops



def _build_noise(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    mapping = {
        "gauss_noise": "GaussNoise",
        "iso_noise": "ISONoise",
        "multiplicative_noise": "MultiplicativeNoise",
        "salt_and_pepper": "SaltAndPepper",
    }
    ops: List[A.BasicTransform] = []
    for key, cls_name in mapping.items():
        spec = cfg.get(key, {})
        if spec.get("enable", False):
            cls = _maybe(cls_name)
            if cls:
                params = _translate_params(key, {k: v for k, v in spec.items() if k != "enable"})
                ops.append(cls(**params))
    return ops


def _build_compression_downscale(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    mapping = {
        "image_compression": "ImageCompression",
        "downscale": "Downscale",
    }
    ops: List[A.BasicTransform] = []
    for key, cls_name in mapping.items():
        spec = cfg.get(key, {})
        if spec.get("enable", False):
            cls = _maybe(cls_name)
            if cls:
                params = _translate_params(key, {k: v for k, v in spec.items() if k != "enable"})
                ops.append(cls(**params))
    return ops


def _build_contrast_enhancement(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    ops: List[A.BasicTransform] = []
    spec = cfg.get("clahe", {})
    if spec.get("enable", False):
        cls = _maybe("CLAHE")
        if cls:
            params = _translate_params("clahe", {k: v for k, v in spec.items() if k != "enable"})
            ops.append(cls(**params))
    return ops


def _build_context_independence(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    ops: List[A.BasicTransform] = []
    # Support the new key; optionally keep backward-compat for old 'grid_shuffle'
    for key, cls_name in [
        ("random_grid_shuffle", "RandomGridShuffle"),  # v2 name
        ("grid_shuffle",        "RandomGridShuffle"),  # legacy config key, optional
    ]:
        spec = cfg.get(key, {})
        if spec.get("enable", False):
            cls = _maybe(cls_name)
            if cls:
                params = _translate_params(key, {k: v for k, v in spec.items() if k != "enable"})
                ops.append(cls(**params))
    return ops


def _build_weather_effects(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    mapping = {
        "random_sun_flare": "RandomSunFlare",
        "random_shadow":    "RandomShadow",
        "random_fog":       "RandomFog",
        "random_rain":      "RandomRain",
        "random_snow":      "RandomSnow",
    }
    ops: List[A.BasicTransform] = []
    for key, cls_name in mapping.items():
        spec = cfg.get(key, {})
        if spec.get("enable", False):
            cls = _maybe(cls_name)
            if cls:
                params = _translate_params(key, {k: v for k, v in spec.items() if k != "enable"})
                ops.append(cls(**params))
    return ops


def _build_spectrogram(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    ops: List[A.BasicTransform] = []
    spec = cfg.get("xy_masking", {})
    if spec.get("enable", False):
        cls = _maybe("XYMasking")
        if cls:
            params = _translate_params("xy_masking", {k: v for k, v in spec.items() if k != "enable"})
            ops.append(cls(**params))
    return ops


def _build_domain_adaptation(cfg: Dict[str, Any]) -> List[A.BasicTransform]:
    ops: List[A.BasicTransform] = []
    spec = cfg.get("fda", {})
    if spec.get("enable", False):
        cls = _maybe("FDA")
        if cls:
            params = _translate_params("fda", {k: v for k, v in spec.items() if k != "enable"})
            ops.append(cls(**params))
    spec = cfg.get("histogram_matching", {})
    if spec.get("enable", False):
        cls = _maybe("HistogramMatching")
        if cls:
            params = _translate_params("histogram_matching", {k: v for k, v in spec.items() if k != "enable"})
            ops.append(cls(**params))
    return ops


# ───────────────────────────────
# Public API
# ───────────────────────────────

def _build_spatial_block(t_spatial: Any, H: int, W: int) -> List[A.BasicTransform]:
    """
    Build the SPATIAL section (image+mask). Returns a list of subgroup-wrapped blocks.
    """
    blocks: List[A.BasicTransform] = []

    if not getattr(t_spatial, "enable", False):
        return blocks

    # A) initial_resize_crop
    sub = getattr(t_spatial, "initial_resize_crop", None)
    if sub and getattr(sub.mode, "enable", False):
        candidates = _build_initial_resize_crop(sub, H, W)
        blk = _wrap_mode(candidates, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # B) basic_geometric
    sub = getattr(t_spatial, "basic_geometric", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_basic_geometric(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # C) affine_perspective
    sub = getattr(t_spatial, "affine_perspective", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_affine_perspective(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # D) dropout_occlusion
    sub = getattr(t_spatial, "dropout_occlusion", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_dropout_occlusion(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # E) context_independence (optional; careful for segmentation)
    sub = getattr(t_spatial, "context_independence", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_context_independence(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    return blocks


def _build_pixel_block(t_pixel: Any) -> List[A.BasicTransform]:
    """
    Build the PIXEL section (image-only). Returns a list of subgroup-wrapped blocks.
    """
    blocks: List[A.BasicTransform] = []

    if not getattr(t_pixel, "enable", False):
        return blocks

    # F) color_space_reduction
    sub = getattr(t_pixel, "color_space_reduction", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_color_space_reduction(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # G) color_augmentations
    sub = getattr(t_pixel, "color_augmentations", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_color_augmentations(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # H) blur
    sub = getattr(t_pixel, "blur", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_blur(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # I) noise
    sub = getattr(t_pixel, "noise", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_noise(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # J) compression_downscale
    sub = getattr(t_pixel, "compression_downscale", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_compression_downscale(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # K) contrast_enhancement
    sub = getattr(t_pixel, "contrast_enhancement", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_contrast_enhancement(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # L) weather_effects
    sub = getattr(t_pixel, "weather_effects", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_weather_effects(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # M) spectrogram (optional)
    sub = getattr(t_pixel, "spectrogram", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_spectrogram(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    # N) domain_adaptation (requires reference images; keep disabled until wired)
    sub = getattr(t_pixel, "domain_adaptation", None)
    if sub and getattr(sub.mode, "enable", False):
        ops = _build_domain_adaptation(sub)
        blk = _wrap_mode(ops, dict(sub.mode))
        if blk:
            blocks.append(blk)

    return blocks

def get_train_transforms(cfg) -> A.ReplayCompose:
    """
    Build the full training augmentation pipeline from the grouped config:
      - Spatial sub-groups (image + mask) with their own modes
      - Pixel sub-groups (image only) with their own modes
      - Final resize to target size
      - ToTensorV2
      - Replay enabled for introspection
    """
    t = cfg.augment.train
    H = int(t.img_size.height)
    W = int(t.img_size.width)

    spatial_blocks = _build_spatial_block(t.spatial, H, W) if getattr(t, "spatial", None) else []
    pixel_blocks   = _build_pixel_block(t.pixel) if getattr(t, "pixel", None) else []

    pipeline: List[A.BasicTransform] = []
    pipeline.extend(spatial_blocks)
    pipeline.extend(pixel_blocks)

    # Ensure fixed output dimensions for batching
    pipeline.append(A.Resize(height=H, width=W, p=1.0))

    # Convert to tensor
    pipeline.append(ToTensorV2())

    return A.ReplayCompose(
        transforms=pipeline,
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(cfg) -> A.Compose:
    """
    Validation pipeline using the same builder, but usually most groups are disabled
    in the val config. Always finishes with ToTensorV2().
    """
    t = cfg.augment.val
    if not t.enable:
        return A.Compose([ToTensorV2()], additional_targets={"mask": "mask"})

    H = int(t.img_size.height)
    W = int(t.img_size.width)

    spatial_blocks = _build_spatial_block(t.spatial, H, W) if getattr(t, "spatial", None) else []
    pixel_blocks   = _build_pixel_block(t.pixel) if getattr(t, "pixel", None) else []

    pipeline: List[A.BasicTransform] = []
    pipeline.extend(spatial_blocks)
    pipeline.extend(pixel_blocks)
    pipeline.append(A.PadIfNeeded(min_height=H, min_width=W, p=1.0))
    pipeline.append(ToTensorV2())

    return A.Compose(
        transforms=pipeline,
        additional_targets={"mask": "mask"},
    )


def get_test_transforms(cfg) -> A.Compose:
    """
    Test transforms: usually same as validation.
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


# ───────────────────────────────
# Custom dual op: apply to image and mask
# ───────────────────────────────

class CoarseDropoutDual(A.CoarseDropout):
    """
    CoarseDropout for v2.x that also applies to masks.
    Strips unsupported args and applies mask_fill_value separately.
    """
    def __init__(self, *args, **kwargs):
        mask_fill_value = kwargs.pop("mask_fill_value", 0)

        # strip args removed in v2
        for k in ["fill_value", "max_holes", "max_height", "max_width"]:
            kwargs.pop(k, None)

        super().__init__(*args, **kwargs)
        self.mask_fill_value = mask_fill_value

    def apply_to_mask(self, mask, **params):
        params = dict(params)
        params["fill_value"] = self.mask_fill_value
        return self.apply(mask, **params)
