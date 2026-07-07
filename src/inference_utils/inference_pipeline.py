# src/inference_utils/inference_pipeline.py

from __future__ import annotations

import glob, json, logging
from pathlib import Path
from typing import Optional, Tuple, Any, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.inference_utils.weight_loader import load_state_dict_flex

log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- helpers -------------------------

def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _to_tensor01(rgb: np.ndarray) -> torch.Tensor:
    # [H,W,3] uint8 -> [1,3,H,W] float32 in [0..1]
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)

def _pad_to_divisor(x: torch.Tensor, divisor: Optional[int]) -> tuple[torch.Tensor, Tuple[int,int,int,int]]:
    if not divisor or divisor <= 1:
        return x, (0, 0, 0, 0)
    _, _, h, w = x.shape
    pad_h = (divisor - (h % divisor)) % divisor
    pad_w = (divisor - (w % divisor)) % divisor
    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    x_pad = F.pad(x, pad, mode="constant", value=0.0)
    return x_pad, (pad[2], pad[3], pad[0], pad[1])  # (top, bottom, left, right)

def _unpad(np_img: np.ndarray, pads: Tuple[int,int,int,int]) -> np.ndarray:
    t, b, l, r = pads
    H, W = np_img.shape[:2]
    return np_img[t:H - b if b > 0 else H, l:W - r if r > 0 else W]


def _normalize_if_configured(x01: torch.Tensor, norm_cfg) -> torch.Tensor:
    """
    Dataset-wide normalization at inference (optional).
    Expects a JSON with {"mean":[...], "std":[...]} in cfg.inference.normalization.stats_path
    """
    if not norm_cfg or not getattr(norm_cfg, "enable", False):
        return x01
    stats_path = Path(norm_cfg.stats_path)
    with open(stats_path, "r") as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=x01.device).view(1, -1, 1, 1)
    std  = torch.tensor(stats["std"],  dtype=torch.float32, device=x01.device).view(1, -1, 1, 1)
    return (x01 - mean) / (std + 1e-6)

def _build_smp_from_cfg(cfg: DictConfig) -> torch.nn.Module:
    kwargs = {
        "arch":            cfg.model.arch_name,
        "encoder_name":    cfg.model.encoder_name,
        "encoder_weights": cfg.model.encoder_weights,
        "in_channels":     cfg.model.in_channels,
        "classes":         cfg.model.classes,
    }
    if getattr(cfg.model, "decoder_attention_type", None):
        kwargs["decoder_attention_type"] = cfg.model.decoder_attention_type
    if getattr(cfg.model, "encoder_freeze", False):
        kwargs["encoder_freeze"] = True
    model = smp.create_model(**kwargs)
    return model.to(DEVICE).eval()

def _overlay_rgb_mask(rgb: np.ndarray, mask_bin: np.ndarray, color=(0, 255, 0), alpha=0.6) -> np.ndarray:
    """rgb uint8 [H,W,3], mask_bin {0,1} [H,W] -> overlay rgb uint8"""
    m = (mask_bin > 0).astype(np.uint8)
    overlay = np.zeros_like(rgb, dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    colored = cv2.bitwise_and(overlay, overlay, mask=m * 255)
    return cv2.addWeighted(rgb, 1.0, colored, alpha, 0.0)

def _save_triptych(rgb: np.ndarray, mask_u8: np.ndarray, overlay: np.ndarray, out_path: Path, max_side: int = 1200) -> None:
    """Horiz concat: RGB | gray mask | overlay. Saved as PNG."""
    def _resize_keep(img: np.ndarray, side: int) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= side:
            return img
        s = side / float(max(h, w))
        nh, nw = int(round(h * s)), int(round(w * s))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    rgb_r = _resize_keep(rgb, max_side)
    mask_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)
    mask_r = _resize_keep(mask_rgb, max_side)
    overlay_r = _resize_keep(overlay, max_side)

    H = max(rgb_r.shape[0], mask_r.shape[0], overlay_r.shape[0])
    def _pad_h(img: np.ndarray, H: int) -> np.ndarray:
        pad = H - img.shape[0]
        if pad <= 0: return img
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    rgb_r, mask_r, overlay_r = _pad_h(rgb_r, H), _pad_h(mask_r, H), _pad_h(overlay_r, H)
    panel = np.concatenate([rgb_r, mask_r, overlay_r], axis=1)
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────
# vegetation cutout helpers
# ─────────────────────────────────────────────────────────────

def _clean_disconnected_mask(bin_mask: np.ndarray, max_gap_px: float) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    # Used only to decide connectivity/grouping — breaks 1-2px noise bridges
    # without affecting which original pixels end up in the final mask.
    label_src = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(label_src, connectivity=4)
    if num <= 2:
        return bin_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    main_mask = (labels == largest_label).astype(np.uint8)
    dist_to_main = cv2.distanceTransform(1 - main_mask, cv2.DIST_L2, 5)

    # Re-label the ORIGINAL (unopened) mask so we don't lose real pixels
    # that the opening removed, then decide per-original-component using
    # distance from the (opened) main_mask.
    num_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=4)
    cleaned = np.zeros_like(bin_mask)
    for label in range(1, num_o):
        comp_pixels = (labels_o == label)
        min_dist = dist_to_main[comp_pixels].min()
        if min_dist <= max_gap_px:
            cleaned[comp_pixels] = 1
    return cleaned


def _tight_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Tight bounding box (x1, y1, x2, y2) — end-exclusive — around all non-zero
    pixels of a crop-space binary mask. Returns None if mask has no foreground.
    """
    bin_mask = mask > 0
    rows = np.any(bin_mask, axis=1)
    cols = np.any(bin_mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    y1, y2 = int(y_idx[0]), int(y_idx[-1]) + 1
    x1, x2 = int(x_idx[0]), int(x_idx[-1]) + 1
    return x1, y1, x2, y2


def _save_metadata_json(out_path: Path, image_name: str, meta: dict) -> None:
    payload = {"image": image_name, **meta}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

def _save_vegetation_cutout(
    seg_cfg: DictConfig,
    crop_rgb: np.ndarray,
    mask_crop: np.ndarray,
    out_path: Path,
    crop_origin_xy: Tuple[int, int],
    detection_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    detection_bbox_padded: bool = False,
    detection_pad_px: int = 0,
) -> dict:
    bin_mask = (mask_crop > 0).astype(np.uint8)

    speckles_removed = False
    if getattr(seg_cfg.clean_disconnected_mask, "enable", False):
        pad_gap_px = getattr(seg_cfg.clean_disconnected_mask, "pad_px", 500)
        cleaned = _clean_disconnected_mask(bin_mask, max_gap_px=pad_gap_px)
        speckles_removed = bool(np.any(cleaned != bin_mask))
        bin_mask = cleaned

    mask_3c    = np.repeat(bin_mask[:, :, np.newaxis], 3, axis=2)
    cutout_rgb = crop_rgb * mask_3c

    ox, oy = crop_origin_xy
    crop_h, crop_w = crop_rgb.shape[:2]
    current_bbox_full = detection_bbox_xyxy or (ox, oy, ox + crop_w, oy + crop_h)

    final_bbox_full = current_bbox_full
    changed = False

    tight = _tight_bbox_from_mask(bin_mask)
    if tight is not None:
        tx1, ty1, tx2, ty2 = tight
        if (tx1, ty1, tx2, ty2) != (0, 0, crop_w, crop_h):
            cutout_rgb = cutout_rgb[ty1:ty2, tx1:tx2]
            final_bbox_full = (ox + tx1, oy + ty1, ox + tx2, oy + ty2)
            changed = True

    cv2.imwrite(str(out_path), cv2.cvtColor(cutout_rgb, cv2.COLOR_RGB2BGR))

    # Padding is only a meaningful concept when a detection actually
    # produced the crop — with no detection, "padded" is inapplicable
    # rather than false, so keep it None to avoid implying a detection
    # existed.
    if detection_bbox_xyxy is None:
        bbox_padded_field = None
        pad_px_field = None
    else:
        bbox_padded_field = detection_bbox_padded
        pad_px_field = detection_pad_px if detection_bbox_padded else 0

    return {
        "detection_bbox_xyxy": list(current_bbox_full) if detection_bbox_xyxy else None,
        "detection_bbox_padded": bbox_padded_field,   # ← None | True | False
        "detection_pad_px": pad_px_field,              # ← None | int
        "cutout_bbox_xyxy": list(final_bbox_full),
        "cutout_bbox_changed": changed,
        "mask_cleaned": speckles_removed,
    }


def _detect_roi_if_enabled(
    roi_cfg,
    rgb: np.ndarray,
    yolo_model: Any = None,
) -> Optional[Tuple[int,int,int,int]]:
    """Run YOLO-based ROI detection on a full RGB frame and return one box.

    Returns ``(x1, y1, x2, y2)`` in integer pixel coords, or ``None`` when
    ROI detection is disabled in config or yields no detections.

    Args:
        yolo_model: Pre-loaded YOLO instance. When provided the model is not
            reloaded from disk, which is critical for per-image loop performance.
    """
    # Skip entirely when ROI detection is not configured or explicitly disabled.
    if not roi_cfg or not getattr(roi_cfg, "enable", False):
        return None

    # Lazy import — only required when ROI detection is active.
    from ultralytics import YOLO

    def _to_f32(obj: Any) -> np.ndarray:
        """Normalise a torch.Tensor or np.ndarray to a float32 ndarray."""
        if isinstance(obj, np.ndarray):
            return obj.astype(np.float32, copy=False)
        return cast(np.ndarray, obj.detach().cpu().numpy()).astype(np.float32, copy=False)

    # Use the pre-loaded model when available; otherwise load from disk.
    # Loading from disk on every call adds significant latency per image.
    yolo = yolo_model if yolo_model is not None else YOLO(roi_cfg.weights)
    # cast keeps Pyright happy; ultralytics returns a list of Results objects.
    # predict() on a single image always returns exactly one Results item.
    res = cast(list[Any], yolo.predict(rgb, verbose=False))

    # Move the first result to CPU before accessing its tensors.
    boxes = getattr(res[0].cpu(), "boxes", None)
    if boxes is None:
        return None

    xyxy_obj = getattr(boxes, "xyxy", None)
    if xyxy_obj is None:
        return None

    xyxy = _to_f32(xyxy_obj)
    if xyxy.shape[0] == 0:
        return None

    # Build a confidence vector aligned with xyxy rows;
    # fall back to uniform 1s when conf is unavailable.
    conf_obj = getattr(boxes, "conf", None)
    conf = np.ones((xyxy.shape[0],), dtype=np.float32) if conf_obj is None else _to_f32(conf_obj)

    # Select the single box to use: largest area or highest confidence.
    pick = str(getattr(roi_cfg, "pick", "best"))
    if pick == "largest":
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(np.argmax(areas))
    else:
        idx = int(np.argmax(conf))

    # .tolist() gives a concrete Python list, avoiding Pyright's
    # "Never is not iterable" false positive on ndarray row unpacking.
    x1, y1, x2, y2 = map(float, xyxy[idx].tolist())

    # Optionally expand the box by pad_px on every side, clamped to image bounds.
    pad_cfg = getattr(roi_cfg, "detection_padding", {})
    pad_px = int(getattr(pad_cfg, "pad_px", 0))
    if bool(getattr(pad_cfg, "enabled", False)) and pad_px > 0:
        x1 = max(0.0, x1 - pad_px)
        y1 = max(0.0, y1 - pad_px)
        x2 = min(float(rgb.shape[1]), x2 + pad_px)
        y2 = min(float(rgb.shape[0]), y2 + pad_px)

    return (int(x1), int(y1), int(x2), int(y2))

def _predict_mask(model: torch.nn.Module, x: torch.Tensor, thr: float) -> np.ndarray:
    with torch.inference_mode():
        logits = model(x.to(DEVICE))
        prob = torch.sigmoid(logits)
    mask = (prob > thr).float()
    return mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

# --- tiling helpers ---

def _hann2d(h, w):
    wx = np.hanning(w)
    wy = np.hanning(h)
    w2d = np.outer(wy, wx)
    w2d = w2d / (w2d.max() + 1e-8)
    return w2d.astype(np.float32)

def _gaussian2d(h, w, sigma_rel=0.3):
    cy, cx = h / 2.0, w / 2.0
    sigma_y = sigma_rel * h
    sigma_x = sigma_rel * w
    ys = np.arange(h)
    xs = np.arange(w)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))
    g = g / (g.max() + 1e-8)
    return g.astype(np.float32)


def _predict_mask_tiled_rgb(
    model: torch.nn.Module,
    img_rgb: np.ndarray,
    norm_cfg=None,
    tile_size: int = 1024,
    overlap: int = 128,
    divisor: Optional[int] = 32,
    thr: float = 0.5,
    blend_method: str = "hann",
    blend_on: str = "prob",
    gaussian_sigma_rel: float = 0.3,
    use_amp: bool = True,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    step = tile_size - overlap
    use_max = blend_method.lower() == "max"

    if use_max:
        fused = np.zeros((H, W), dtype=np.float32)
    else:
        acc  = np.zeros((H, W), dtype=np.float32)
        wsum = np.zeros((H, W), dtype=np.float32)

    y = 0
    while y < H:
        x = 0
        while x < W:
            y1c, y2c = y, min(y + tile_size, H)
            x1c, x2c = x, min(x + tile_size, W)
            tile = img_rgb[y1c:y2c, x1c:x2c]
            th, tw = tile.shape[:2]

            tile_t = _to_tensor01(tile)
            tile_pad, pads = _pad_to_divisor(tile_t, divisor)
            tile_norm = _normalize_if_configured(tile_pad, norm_cfg)

            with torch.inference_mode():
                if use_amp and DEVICE == "cuda":
                    with torch.autocast(device_type="cuda"):
                        logits = model(tile_norm.to(DEVICE))
                        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
                else:
                    logits = model(tile_norm.to(DEVICE))
                    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
            if divisor:
                probs = _unpad(probs, pads)

            tile_q = probs if blend_on.lower() == "prob" else (probs >= thr).astype(np.float32)

            if use_max:
                fused[y1c:y2c, x1c:x2c] = np.maximum(fused[y1c:y2c, x1c:x2c], tile_q)
            else:
                if blend_method.lower() == "gaussian":
                    win = _gaussian2d(th, tw, gaussian_sigma_rel)
                else:
                    win = _hann2d(th, tw)
                win = cv2.resize(win, (tile_q.shape[1], tile_q.shape[0]), interpolation=cv2.INTER_LINEAR)
                acc[y1c:y2c, x1c:x2c]  += tile_q * win
                wsum[y1c:y2c, x1c:x2c] += win

            x += step
        y += step

    out = fused if use_max else (acc / np.clip(wsum, 1e-6, None))
    return (out >= thr).astype(np.uint8)


# ------------------------- pipeline -------------------------

def run_inference_pipeline(cfg: DictConfig) -> None:
    """
    Local inference runner:
      - reads images (cfg.inference.input_dir or paths.test_images_dir/val_images_dir)
      - optional ROI detection (YOLO)
      - segmentation with SMP model from cfg.model
      - saves: raw masks, overlays, triptych (RGB | mask | overlay), and vegetation cutouts
      - stores each run under a timestamped subfolder inside the Hydra run dir
      - optionally logs previews to Weights & Biases
    """
    base_run_dir = Path(HydraConfig.get().runtime.output_dir)

    # versioned subdir (keeps previous behavior)
    stamp = f"version_{cfg.job.job_now_date}_{cfg.job.job_now_time}"
    run_dir = base_run_dir / stamp
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (run_dir / "triptych").mkdir(parents=True, exist_ok=True)
    (run_dir / "cutouts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)

    # W&B (optional)
    wb_cfg = getattr(getattr(cfg, "inference", None), "logger", {}).get("wandb", {})
    use_wandb = bool(wb_cfg.get("enable", False))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wb_cfg.get("project", cfg.project.name),
                entity=wb_cfg.get("entity", None),
                name=wb_cfg.get("run_name", stamp),
                dir=str(run_dir),
                config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
                save_code=False,
                reinit=True,
            )
            log.info("[inference] W&B logging enabled.")
        except Exception as e:
            use_wandb = False
            log.warning(f"[inference] W&B init failed: {e}")

    # model
    model = _build_smp_from_cfg(cfg)

    # weights
    weights_path = Path(cfg.inference.seg.weights_path)
    if "*" in str(weights_path):
        matches = sorted(glob.glob(str(weights_path)))
        if not matches:
            raise FileNotFoundError(f"No weights matched pattern: {weights_path}")
        weights_path = Path(matches[-1])
    missing, unexpected = load_state_dict_flex(model, weights_path, strict=False)
    if missing or unexpected:
        log.warning(f"[inference] load_state_dict: missing={missing}, unexpected={unexpected}")
    log.info(f"[inference] loaded weights from: {weights_path}")

    # params
    thr      = float(getattr(cfg.inference.seg, "threshold", 0.5))
    divisor  = int(getattr(cfg.inference.seg, "pad_to_divisor", 0)) or None
    alpha    = float(getattr(cfg.inference.overlay, "alpha", 0.6))
    color    = tuple(int(c) for c in getattr(cfg.inference.overlay, "color", [0, 255, 0]))
    max_side = int(getattr(cfg.inference, "preview_max_side", 1200))

    save_cfg     = getattr(cfg.inference, "save", {})
    save_triptych = bool(getattr(save_cfg, "triptych", True))
    save_overlay  = bool(getattr(save_cfg, "overlay",  True))
    save_mask     = bool(getattr(save_cfg, "raw_mask", True))
    save_cutout   = bool(getattr(save_cfg, "cutout",   False))
    save_metadata = bool(getattr(save_cfg, "metadata", False))
    roi_cfg = getattr(cfg.inference, "roi", None)
    seg_cfg = getattr(cfg.inference, "seg", None)


    # Pre-load YOLO ROI model once — reused for every image in the loop.
    # Loading the model inside the loop would reload weights from disk each iteration.
    _roi_yolo = None
    if roi_cfg and getattr(roi_cfg, "enable", False):
        from ultralytics import YOLO as _YOLO
        _roi_yolo = _YOLO(roi_cfg.weights)
        log.info(f"[inference] ROI model loaded: {roi_cfg.weights}")

    # normalization (dataset-wide) if requested
    norm_cfg = getattr(cfg.inference, "normalization", None)

    # ---------------- inputs ----------------
    # Priority: cfg.inference.input_dir -> paths.test_images_dir -> paths.val_images_dir
    in_dir = getattr(cfg.inference, "input_dir", None)
    if in_dir:
        img_root = Path(in_dir)
    else:
        p = Path(cfg.paths.test_images_dir)
        img_root = p if p.exists() else Path(cfg.paths.val_images_dir)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    imgs = []
    for e in exts:
        imgs.extend(sorted(img_root.rglob(e)))
    if not imgs:
        log.warning(f"[inference] no images found under: {img_root}")
        return
    
    # main loop
    for ip in imgs:
        rgb = _read_rgb(ip)
        H, W = rgb.shape[:2]

        roi = _detect_roi_if_enabled(roi_cfg, rgb, yolo_model=_roi_yolo)
        if roi is None:
            log.warning(f"[inference] no ROI detected for {ip.name}; segmenting full image.")
            x1, y1, x2, y2 = 0, 0, W, H
        else:
            x1, y1, x2, y2 = roi
        pad_cfg = getattr(roi_cfg, "detection_padding", {}) if roi_cfg else {}
        pad_px = int(getattr(pad_cfg, "pad_px", 0))
        bbox_was_padded = bool(
            roi is not None and getattr(pad_cfg, "enabled", False) and pad_px > 0
        )

        crop = rgb[y1:y2, x1:x2].copy()

        # tiling toggle
        tile_cfg   = getattr(cfg.inference.seg, "tile", None)
        use_tiling = bool(tile_cfg and getattr(tile_cfg, "enable", False))

        if use_tiling:
            tsize     = int(getattr(tile_cfg, "tile_size", 2048))
            tover     = int(getattr(tile_cfg, "overlap", 256))
            blend_cfg = getattr(tile_cfg, "blend", {})
            b_method  = str(getattr(blend_cfg, "method", "hann"))
            b_on      = str(getattr(blend_cfg, "on", "prob"))
            b_sigma   = float(getattr(blend_cfg, "sigma", 0.3))
            b_amp     = bool(getattr(tile_cfg, "amp", {}).get("enable", True))

            mask_crop = _predict_mask_tiled_rgb(
                model,
                crop,
                norm_cfg=norm_cfg,
                tile_size=tsize,
                overlap=tover,
                divisor=divisor,
                thr=thr,
                blend_method=b_method,
                blend_on=b_on,
                gaussian_sigma_rel=b_sigma,
                use_amp=b_amp,
            )
        else:
            x01 = _to_tensor01(crop)
            x01_pad, pads = _pad_to_divisor(x01, divisor)
            x_in = _normalize_if_configured(x01_pad, norm_cfg)
            mask_small = _predict_mask(model, x_in, thr=thr)
            mask_unpad = _unpad(mask_small, pads) if divisor else mask_small
            mask_crop  = cv2.resize(mask_unpad, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

        # paste back to canvas
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_crop

        # visuals
        overlay = _overlay_rgb_mask(rgb, full_mask, color=color, alpha=alpha)
        mask_u8 = (full_mask * 255).astype(np.uint8)

        stem = ip.stem
        if save_mask:
            cv2.imwrite(str(run_dir / "masks" / f"{stem}.png"), mask_u8)
        if save_overlay:
            cv2.imwrite(str(run_dir / "overlays" / f"{stem}_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if save_triptych:
            _save_triptych(rgb, mask_u8, overlay, run_dir / "triptych" / f"{stem}_triptych.png",
                           max_side=max_side)

        # ── vegetation cutout ──────────────────────────────────────
        if save_cutout:
            assert seg_cfg is not None, "seg config must be provided to save cutouts"
            cutout_meta = _save_vegetation_cutout(
                seg_cfg=seg_cfg,
                crop_rgb=crop,
                mask_crop=mask_crop,
                out_path=run_dir / "cutouts" / f"{stem}_cutout.png",
                crop_origin_xy=(x1, y1),
                detection_bbox_xyxy=roi,
                detection_bbox_padded=bbox_was_padded,
                detection_pad_px=pad_px
            )
            if save_metadata:
                _save_metadata_json(
                    run_dir / "metadata" / f"{stem}.json",
                    image_name=ip.name,
                    meta=cutout_meta,
                )
        # ───────────────────────────────────────────────────────────────

        # per-image W&B logging
        if use_wandb:
            try:
                import wandb
                to_log = {}
                if save_mask:
                    to_log["inference/mask"] = wandb.Image(str(run_dir / "masks" / f"{stem}.png"))
                if save_overlay:
                    to_log["inference/overlay"] = wandb.Image(str(run_dir / "overlays" / f"{stem}_overlay.png"))
                if save_triptych:
                    to_log["inference/triptych"] = wandb.Image(str(run_dir / "triptych" / f"{stem}_triptych.png"))
                if save_cutout:                                              # ← NEW
                    to_log["inference/cutout"] = wandb.Image(               # ← NEW
                        str(run_dir / "cutouts" / f"{stem}_cutout.png")     # ← NEW
                    )                                                        # ← NEW
                if to_log:
                    wandb.log(to_log)
            except Exception as e:
                log.debug(f"W&B per-image log failed for {ip.name}: {e}")

        log.info(f"[inference] ✔ {ip.name}")

    log.info(f"[inference] done. Saved to: {run_dir}")
