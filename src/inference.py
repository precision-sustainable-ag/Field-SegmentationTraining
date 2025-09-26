# src/inference.py

from __future__ import annotations

import logging, json, glob
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
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


# ----------------------------- main entry -----------------------------

def inference(cfg: DictConfig) -> None:
    """
    Local inference runner:
      - reads images (cfg.inference.input_dir or paths.test_images_dir/val_images_dir)
      - optional ROI detection (YOLO)
      - segmentation with SMP model from cfg.model
      - saves: raw masks, overlays, and triptych (RGB | mask | overlay)
      - writes under Hydra run dir (same version folder as the rest)
    """
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (run_dir / "triptych").mkdir(parents=True, exist_ok=True)

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
    for ip in imgs:
        rgb = _read_rgb(ip)
        H, W = rgb.shape[:2]

        # ROI
        roi = _detect_roi_if_enabled(cfg, rgb)
        if roi is None:
            x1, y1, x2, y2 = 0, 0, W, H
        else:
            x1, y1, x2, y2 = roi

        crop = rgb[y1:y2, x1:x2].copy()
        x01 = _to_tensor01(crop)
        x01, pads = _pad_to_divisor(x01, divisor)
        x_in = _normalize_if_configured(x01, norm_cfg)

        # predict
        mask_small = _predict_mask(model, x_in, thr=thr)
        mask_unpad = _unpad(mask_small, pads) if divisor else mask_small
        mask_resized = cv2.resize(mask_unpad, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

        # paste back to full-res canvas
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_resized

        # visuals
        overlay = _overlay_rgb_mask(rgb, full_mask, color=color, alpha=alpha)
        mask_u8 = (full_mask * 255).astype(np.uint8)

        stem = ip.stem
        cv2.imwrite(str(run_dir / "masks" / f"{stem}.png"), mask_u8)
        cv2.imwrite(str(run_dir / "overlays" / f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        _save_triptych(rgb, mask_u8, overlay, run_dir / "triptych" / f"{stem}_triptych.png", max_side=max_side)

        log.info(f"[inference] ✔ {ip.name}")

    log.info("[inference] done.")
