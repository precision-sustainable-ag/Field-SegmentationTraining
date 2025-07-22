# src/preprocess_utils/data_stats.py

import json
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

def _stats_for_image(img_path: Path):
    """
    Compute per-image channel statistics.

    Args:
        img_path (Path): Path to a single RGB image.

    Returns:
        tuple:
            sum_c (np.ndarray[3]): Sum of pixel values per channel.
            sum_sq (np.ndarray[3]): Sum of squared pixel values per channel.
            pixels (int): Total number of pixels in the image.
    """
    # Load image and convert to RGB array in [0,1]
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0  # shape = (H, W, 3)

    h, w, _ = arr.shape
    pixels = h * w

    # Flatten to (pixels, 3) for easy channel-wise sum
    flat = arr.reshape(pixels, 3)

    # Compute sum of values and sum of squares for each of the 3 channels
    sum_c  = flat.sum(axis=0)        # ∑ pixel values per channel
    sum_sq = (flat ** 2).sum(axis=0) # ∑ squared pixel values per channel

    return sum_c, sum_sq, pixels

def compute_rgb_mean_std(
    cfg: any
) -> None:
    """
    Compute dataset-wide mean & std for all training RGB images,
    and write to ${paths.project_datastats_dir}/rgb_mean_std.json.

    Uses optional multiprocessing based on cfg.preprocess.compute_data_stats.

    Args:
        cfg: full Hydra config; expects:
          - cfg.paths.train_images_dir
          - cfg.paths.project_datastats_dir
          - cfg.preprocess.compute_data_stats.use_concurrency
          - cfg.preprocess.compute_data_stats.num_workers
    """
    # 1) gather inputs
    images_dir = Path(cfg.paths.train_images_dir)
    out_dir    = Path(cfg.paths.project_datastats_dir)
    out_file   = out_dir / "rgb_mean_std.json"
    img_paths  = sorted(images_dir.glob("*"))

    cc_cfg     = cfg.preprocess.compute_data_stats
    use_cc     = bool(cc_cfg.use_concurrency)
    workers    = int(cc_cfg.num_workers) if use_cc else 1

    sum_c        = np.zeros(3, dtype=np.float64)
    sum_sq       = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    if use_cc and workers > 1:
        # ─── Parallel processing ────────────────────────────────────────────
        with ProcessPoolExecutor(max_workers=workers) as exe:
            # Submit one task per image
            futures = {exe.submit(_stats_for_image, p): p for p in img_paths}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    sc, ssq, pix = fut.result()
                    sum_c        += sc
                    sum_sq       += ssq
                    total_pixels += pix
                except Exception as e:
                    log.error(f"[data_stats] {p.name} failed: {e}")
    else:
        # ─── Sequential fallback ────────────────────────────────────────────
        for p in img_paths:
            try:
                sc, ssq, pix = _stats_for_image(p)
                sum_c        += sc
                sum_sq       += ssq
                total_pixels += pix
            except Exception as e:
                log.error(f"[data_stats] {p.name} failed: {e}")

    # ─── Finalize mean & standard deviation ───────────────────────────────
    # mean per channel = sum_c / total_pixels
    mean = (sum_c / total_pixels).tolist()

    # variance per channel = E[x^2] - (E[x])^2
    var = (sum_sq / total_pixels) - np.square(mean)
    std = np.sqrt(var).tolist()

    # ─── Write to JSON ─────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=4)
