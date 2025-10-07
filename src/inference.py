# src/inference.py

from __future__ import annotations

import logging, json, glob
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import segmentation_models_pytorch as smp

# your flexible loader
from src.inference_utils.weight_loader import load_state_dict_flex

log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- small helpers -------------------------

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
    # pad = (left, right, top, bottom)
    pad = (0, pad_w, 0, pad_h)
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
    # cv2 wants BGR:
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

def _detect_roi_if_enabled(cfg, rgb: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    roi_cfg = getattr(cfg.inference, "roi", None)
    if not roi_cfg or not getattr(roi_cfg, "enable", False):
        return None
    from ultralytics import YOLO
    yolo = YOLO(roi_cfg.weights)
    res = yolo.predict(rgb, verbose=False)
    boxes = res[0].boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    pick = str(getattr(roi_cfg, "pick", "best"))
    if pick == "largest":
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(np.argmax(areas))
    else:
        idx = int(np.argmax(conf))
    x1, y1, x2, y2 = xyxy[idx]
    return (int(x1), int(y1), int(x2), int(y2))

def _predict_mask(model: torch.nn.Module, x: torch.Tensor, thr: float) -> np.ndarray:
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        prob = torch.sigmoid(logits)
    mask = (prob > thr).float()
    return mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

def _hann2d(h, w):
    wx = np.hanning(w)
    wy = np.hanning(h)
    w2d = np.outer(wy, wx)
    w2d = w2d / (w2d.max() + 1e-8)
    return w2d.astype(np.float32)

def _gaussian2d(h, w, sigma_rel=0.3):
    # sigma as a fraction of tile size (rough, but works well)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dy2 = (yy - cy) ** 2
    dx2 = (xx - cx) ** 2
    sigma_y = max(1.0, sigma_rel * h)
    sigma_x = max(1.0, sigma_rel * w)
    w2d = np.exp(-0.5 * (dy2 / (sigma_y ** 2) + dx2 / (sigma_x ** 2)))
    w2d = w2d / (w2d.max() + 1e-8)
    return w2d.astype(np.float32)

def _tile_blend_window(h, w, method: str, sigma_rel: float) -> np.ndarray:
    m = (method or "hann").lower()
    if m == "hann":
        return _hann2d(h, w)
    if m == "gaussian":
        return _gaussian2d(h, w, sigma_rel=sigma_rel)
    if m == "uniform":
        return np.ones((h, w), dtype=np.float32)
    if m == "max":
        # we won’t use a window in MAX mode
        return np.ones((h, w), dtype=np.float32)
    # fallback
    return _hann2d(h, w)

def _predict_mask_tiled_rgb(
    model,
    rgb: np.ndarray,      # ROI or full image, HxWx3 (uint8/RGB)
    norm_cfg,
    tile_size: int,
    overlap: int,
    divisor: Optional[int],
    thr: float,
    blend_method: str = "hann",    # hann | uniform | gaussian | max
    blend_on: str = "prob",        # prob | bin
    gaussian_sigma_rel: float = 0.3,
) -> np.ndarray:
    """
    Slide-window inference with overlap and selectable blending.
    Returns a binary mask (uint8) of the same HxW as rgb.
    """
    H, W = rgb.shape[:2]
    step = max(1, tile_size - overlap)

    # Accumulators
    if blend_method.lower() == "max":
        # keep running max (work on probs or bin depending on blend_on)
        fused = np.zeros((H, W), dtype=np.float32)
        use_max = True
    else:
        acc = np.zeros((H, W), dtype=np.float32)
        wsum = np.zeros((H, W), dtype=np.float32)
        use_max = False

    y = 0
    while y < H:
        x = 0
        y2 = min(y + tile_size, H)
        y1 = max(0, y2 - tile_size)
        th = y2 - y1

        while x < W:
            x2 = min(x + tile_size, W)
            x1 = max(0, x2 - tile_size)
            tw = x2 - x1

            tile_rgb = rgb[y1:y2, x1:x2, :]
            win = _tile_blend_window(th, tw, blend_method, gaussian_sigma_rel)

            # to tensor [1,3,th,tw]
            tile_x01 = _to_tensor01(tile_rgb)

            # run model on this tile -> prob map
            # (reuse your single path but return PROB, not bin)
            # We'll compute prob here directly to avoid thresholding first:
            tile_xpad, pads = _pad_to_divisor(tile_x01, divisor)
            tile_xin = _normalize_if_configured(tile_xpad, norm_cfg)
            with torch.no_grad():
                logits = model(tile_xin.to(DEVICE))
                probs = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
            if divisor:
                probs = _unpad(probs, pads)  # [th, tw]

            # pick quantity to blend (prob or bin)
            tile_q = probs if blend_on.lower() == "prob" else (probs >= thr).astype(np.float32)

            if use_max:
                # overwrite with max
                fused[y1:y2, x1:x2] = np.maximum(fused[y1:y2, x1:x2], tile_q)
            else:
                acc[y1:y2, x1:x2] += tile_q * win
                wsum[y1:y2, x1:x2] += win

            x += step
        y += step

    if use_max:
        out = fused
    else:
        wsum = np.clip(wsum, 1e-6, None)
        out = acc / wsum

    # final threshold -> binary
    out_bin = (out >= thr).astype(np.uint8)
    return out_bin


# ----------------------------- main entry -----------------------------

def inference(cfg: DictConfig) -> None:
    """
    Local inference runner:
      - reads images (cfg.inference.input_dir or paths.test_images_dir/val_images_dir)
      - optional ROI detection (YOLO)
      - segmentation with SMP model from cfg.model
      - saves: raw masks, overlays, and triptych (RGB | mask | overlay)
      - stores each run under a timestamped subfolder inside the Hydra run dir
      - optionally logs previews to Weights & Biases
    """
    # Hydra job dir (shared with pipeline_log.yaml)
    base_run_dir = Path(HydraConfig.get().runtime.output_dir)

    # Versioned subdir per inference run
    stamp = f"version_{cfg.job.job_now_date}_{cfg.job.job_now_time}"
    run_dir = base_run_dir / stamp
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (run_dir / "triptych").mkdir(parents=True, exist_ok=True)

    # ---------------- W&B (optional) ----------------
    wb_cfg = getattr(getattr(cfg, "inference", None), "logger", {}).get("wandb", {})
    use_wandb = bool(wb_cfg.get("enable", False))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wb_cfg.get("project", cfg.project.name),
                entity=wb_cfg.get("entity", None),
                name=wb_cfg.get("run_name", stamp),
                dir=str(run_dir),  # files under the same versioned folder
                config=OmegaConf.to_container(cfg, resolve=True),
                save_code=False,
                reinit=True,
            )
            log.info("[inference] W&B logging enabled.")
        except Exception as e:
            use_wandb = False
            log.warning(f"[inference] W&B init failed: {e}")

    # ---------------- model ----------------
    model = _build_smp_from_cfg(cfg)

    # Find weights
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

    save_triptych = bool(getattr(getattr(cfg.inference, "save", {}), "triptych", True))
    save_overlay  = bool(getattr(getattr(cfg.inference, "save", {}), "overlay",  True))
    save_mask     = bool(getattr(getattr(cfg.inference, "save", {}), "raw_mask", True))

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

    # ---------------- loop ----------------
    wb_images = []  # collect a few previews for W&B
    for i, ip in enumerate(imgs):
        rgb = _read_rgb(ip)
        H, W = rgb.shape[:2]

        # ROI
        roi = _detect_roi_if_enabled(cfg, rgb)
        if roi is None:
            x1, y1, x2, y2 = 0, 0, W, H
        else:
            x1, y1, x2, y2 = roi

        crop = rgb[y1:y2, x1:x2].copy()

        # predict
        tile_cfg   = getattr(cfg.inference.seg, "tile", None)
        use_tiling = bool(tile_cfg and getattr(tile_cfg, "enable", False))

        if use_tiling:
            tsize     = int(getattr(tile_cfg, "tile_size", 2048))
            tover     = int(getattr(tile_cfg, "overlap", 256))
            blend_cfg = getattr(tile_cfg, "blend", {})
            b_method  = str(getattr(blend_cfg, "method", "hann"))
            b_on      = str(getattr(blend_cfg, "on", "prob"))
            b_sigma   = float(getattr(blend_cfg, "sigma", 0.3))

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
            )
        else:
            # original single-shot path
            x01 = _to_tensor01(crop)
            x01_pad, pads = _pad_to_divisor(x01, divisor)
            x_in = _normalize_if_configured(x01_pad, norm_cfg)
            mask_small = _predict_mask(model, x_in, thr=thr)
            mask_unpad = _unpad(mask_small, pads) if divisor else mask_small
            mask_crop  = cv2.resize(mask_unpad, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

        # paste back to full-res canvas
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_crop

        # visuals
        overlay = _overlay_rgb_mask(rgb, full_mask, color=color, alpha=alpha)
        mask_u8 = (full_mask * 255).astype(np.uint8)

        stem = ip.stem
        if save_mask:
            cv2.imwrite(str(run_dir / "masks" / f"{stem}.png"), mask_u8)
        if save_overlay:
            cv2.imwrite(
                str(run_dir / "overlays" / f"{stem}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
        if save_triptych:
            _save_triptych(
                rgb, mask_u8, overlay,
                run_dir / "triptych" / f"{stem}_triptych.png",
                max_side=max_side,
            )

        # --- per-image W&B logging (after files are saved) ---
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
                if to_log:
                    wandb.log(to_log)
            except Exception as e:
                log.debug(f"W&B per-image log failed for {ip.name}: {e}")

        log.info(f"[inference] ✔ {ip.name}")

    # Flush W&B previews
    if use_wandb and wb_images:
        try:
            import wandb
            wandb.log({"inference/overlays": wb_images})
        except Exception as e:
            log.debug(f"W&B log failed: {e}")

    log.info(f"[inference] done. Saved to: {run_dir}")
