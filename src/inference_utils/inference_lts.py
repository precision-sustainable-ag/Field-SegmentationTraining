# src/inference_utils/inference_lts.py

from __future__ import annotations
import sqlite3
from pathlib import Path
import logging
import pandas as pd

import cv2
import numpy as np
import torch

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.inference_utils.inference_pipeline import (
    _read_rgb,
    _overlay_rgb_mask,
    _save_triptych,
    _to_tensor01,
    _pad_to_divisor,
    _unpad,
    _normalize_if_configured,
    _predict_mask,
    _predict_mask_tiled_rgb,
    _detect_roi_if_enabled,
    _build_smp_from_cfg,
)
from src.inference_utils.weight_loader import load_state_dict_flex

log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
# DB Helpers (all safe with context managers)
# ----------------------------------------------------------

def _load_lts_rows(cfg: DictConfig) -> pd.DataFrame:
    """
    Load candidate LTS images for segmentation inference.

    Applies the following filters:
        • final_mask_path IS NULL            → not yet completed in relabel step
        • developed_image_path IS NOT NULL  → image exists in LTS
        • extension IN ('.jpg', '.JPG')     → exclude RAW (.ARW) files
        • optional plant_type and/or common_name filters from config
        • LIMIT from config (default: 50)

    Additionally:
        • Converts developed_image_path → absolute filesystem paths
        • Removes rows whose file does not exist
        • Deduplicates identical paths (e.g. ARW+JPG duplicates)
    """

    # -------------------------------------------------------------------------
    # 1. Validate DB
    # -------------------------------------------------------------------------
    db_path = Path(cfg.paths.agir_field_db)
    assert db_path.exists(), f"LTS DB missing: {db_path}"

    table = "field_data"

    # -------------------------------------------------------------------------
    # 2. Read filters from config
    # -------------------------------------------------------------------------
    lts_cfg = cfg.inference.lts
    # -------------------------------------------------------------------------
    # Handle "enable" flag:
    # If enable=false → ignore config filters and use defaults.
    # -------------------------------------------------------------------------
    if not getattr(lts_cfg, "enable", True):
        plant_type = None
        common_names = None
        limit = 50  # default fallback
        log.info("[LTS] lts.enable=False → ignoring plant_type/common_name filters.")
    else:
        plant_type = getattr(lts_cfg, "plant_type", None)
        common_names = getattr(lts_cfg, "common_name", None)
        limit = int(getattr(lts_cfg, "limit", 50))

    # Normalize common_names → list[str] or None
    if isinstance(common_names, str):
        common_names = [common_names]
    elif common_names is not None and not isinstance(common_names, list):
        raise ValueError("common_name must be a string or list of strings.")

    # -------------------------------------------------------------------------
    # 3. SQL WHERE builder
    # -------------------------------------------------------------------------
    where_clauses = [
        "final_mask_path IS NULL",
        "developed_image_path IS NOT NULL",
        "extension IN ('.jpg', '.JPG')"    # match DB values with leading dot
    ]
    params: list = []

    # plant_type filter (case-insensitive)
    if plant_type:
        where_clauses.append("LOWER(plant_type) = LOWER(?)")
        params.append(plant_type)

    # common_name filter (case-insensitive, IN list)
    if common_names:
        placeholders = ",".join("?" * len(common_names))
        where_clauses.append(f"LOWER(common_name) IN ({placeholders})")
        params.extend([c.lower() for c in common_names])

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT
            image_id,
            extension,
            common_name,
            plant_type,
            developed_image_path,
            final_mask_path
        FROM {table}
        WHERE {where_sql}
        LIMIT {limit};
    """

    log.info(f"[LTS] SQL: {sql.strip()}  PARAMS={params}")

    # -------------------------------------------------------------------------
    # 4. Execute SQL
    # -------------------------------------------------------------------------
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON;")
        df = pd.read_sql_query(sql, con, params=params)

    # -------------------------------------------------------------------------
    # 5. Convert developed_image_path → absolute filesystem paths
    # -------------------------------------------------------------------------
    base_lts = Path(cfg.paths.longterm_storage)

    def to_abs(p: str | None) -> str | None:
        return str(base_lts / p) if isinstance(p, str) else None

    df["abs_path"] = df["developed_image_path"].apply(to_abs)

    # Keep only rows whose image exists
    df = df[df["abs_path"].apply(lambda p: p is not None and Path(p).exists())]

    # -------------------------------------------------------------------------
    # 6. De-duplicate by absolute path
    #    (prevents ARW+JPG dual entries → duplicate processing)
    # -------------------------------------------------------------------------
    df = df.drop_duplicates(subset=["abs_path"]).reset_index(drop=True)

    log.info(
        f"[LTS] Loaded {len(df)} candidate images from LTS DB "
        f"(filters: {where_sql})"
    )

    return df

# ----------------------------------------------------------
# Main LTS inference
# ----------------------------------------------------------

def run_inference_lts(cfg: DictConfig) -> None:
    """
    LTS-mode inference:
      - Reads list of image paths from LTS SQLite DB
      - Loads images directly from LTS filesystem
      - Runs your exact inference pipeline (ROI → tiling → threshold)
      - Writes output to the same Hydra version folder
    """
    base_run_dir = Path(HydraConfig.get().runtime.output_dir)
    stamp = f"version_{cfg.job.job_now_date}_{cfg.job.job_now_time}"
    run_dir = base_run_dir / stamp

    # Create folders
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (run_dir / "triptych").mkdir(parents=True, exist_ok=True)

    # W&B (optional)
    wb_cfg = cfg.inference.logger.wandb if "logger" in cfg.inference and "wandb" in cfg.inference.logger else {}
    use_wandb = bool(wb_cfg.get("enable", False))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wb_cfg.get("project", cfg.project.name),
                entity=wb_cfg.get("entity", None),
                name=wb_cfg.get("run_name", stamp),
                dir=str(run_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                save_code=False,
                reinit=True,
            )
            log.info("[inference:LTS] W&B logging enabled.")
        except Exception as e:
            log.warning(f"[inference:LTS] W&B init failed: {e}")
            use_wandb = False

    # Load SMP model
    model = _build_smp_from_cfg(cfg)

    # Load weights (same logic as local inference)
    weights_path = Path(cfg.inference.seg.weights_path)
    missing, unexpected = load_state_dict_flex(model, weights_path, strict=False)
    if missing or unexpected:
        log.warning(f"[LTS] load_state_dict: missing={missing}, unexpected={unexpected}")
    log.info(f"[LTS] loaded weights from: {weights_path}")

    # Inference parameters
    thr      = float(cfg.inference.seg.threshold)
    divisor  = int(cfg.inference.seg.pad_to_divisor) or None
    alpha    = float(cfg.inference.overlay.alpha)
    color    = tuple(cfg.inference.overlay.color)
    max_side = int(cfg.inference.preview_max_side)
    tile_cfg = getattr(cfg.inference.seg, "tile", None)
    norm_cfg = getattr(cfg.inference, "normalization", None)

    # Load image list from LTS database
    df = _load_lts_rows(cfg)

    # ---------------- Main loop over LTS images ----------------
    for _, row in df.iterrows():
        img_path = Path(row["abs_path"])
        image_id = row["image_id"]

        try:
            rgb = _read_rgb(img_path)
        except Exception as e:
            log.error(f"[LTS] Failed to read {img_path}: {e}")
            continue

        H, W = rgb.shape[:2]

        roi = _detect_roi_if_enabled(cfg, rgb)
        if roi is None:
            # No ROI → use the full image
            x1, y1, x2, y2 = 0, 0, W, H
            crop = rgb[y1:y2, x1:x2].copy()
        else:
            crop = rgb.copy()
            x1, y1 = 0, 0

        # Tiled vs Single-shot
        if tile_cfg and tile_cfg.enable:
            tsize = tile_cfg.tile_size
            tover = tile_cfg.overlap
            b_method = tile_cfg.blend.method
            b_on = str(tile_cfg.blend.get("on", "prob"))
            b_sigma = tile_cfg.blend.sigma
            b_amp = tile_cfg.amp.enable

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
            xpad, pads = _pad_to_divisor(x01, divisor)
            x_in = _normalize_if_configured(xpad, norm_cfg)
            mask_small = _predict_mask(model, x_in, thr=thr)
            mask_crop = _unpad(mask_small, pads) if divisor else mask_small
            mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Paste into full canvas
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_crop

        overlay = _overlay_rgb_mask(rgb, full_mask, color=color, alpha=alpha)
        mask_u8 = (full_mask * 255).astype(np.uint8)

        # Save
        stem = img_path.stem
        cv2.imwrite(str(run_dir / "masks" / f"{stem}.png"), mask_u8)
        cv2.imwrite(str(run_dir / "overlays" / f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        _save_triptych(rgb, mask_u8, overlay,
                       run_dir / "triptych" / f"{stem}_triptych.png",
                       max_side=max_side)

        # W&B per-image logging
        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "inference_LTS/mask": wandb.Image(str(run_dir / "masks" / f"{stem}.png")),
                    "inference_LTS/overlay": wandb.Image(str(run_dir / "overlays" / f"{stem}_overlay.png")),
                    "inference_LTS/triptych": wandb.Image(str(run_dir / "triptych" / f"{stem}_triptych.png")),
                })
            except Exception as e:
                log.debug(f"[LTS] W&B log failed: {e}")

        log.info(f"[LTS] ✔ {img_path.name}")

    log.info(f"[LTS] Done. Saved to: {run_dir}")
