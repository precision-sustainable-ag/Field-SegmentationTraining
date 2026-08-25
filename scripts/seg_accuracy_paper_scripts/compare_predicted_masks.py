"""Compare binary segmentation masks produced by multiple models.

This script evaluates every model directory under :data:`PRED_DIR` against the
PNG ground-truth masks in :data:`GT_DIR`. It can produce:

* one row of metrics for each matched image/model pair;
* mean metrics for each model;
* metrics split by categories in ``balanced_groupings_test_images.json``;
* bar charts comparing model-level mean metrics;
* optional grids of raw masks and optional overlays on the source photographs.

Expected input layout::

    BASE_DIR/
    ├── gt_masks/
    │   ├── image_001.png
    │   └── ...
    ├── images/
    │   ├── image_001.jpg
    │   └── ...
    └── predict_masks/
        ├── model_a/
        │   ├── image_001.png
        │   └── ...
        └── model_b/
            └── ...

Adjust the paths and feature switches, then run::

    python scripts/compare_predicted_masks.py

All outputs are written beneath :data:`OUT_DIR`, which defaults to
``BASE_DIR/mask_comparison_metrics``. The enabled pipeline switches determine
which artifacts are created::

    OUT_DIR/
    ├── per_image_metrics.csv                 # CALCULATE_METRICS
    ├── summary_metrics.csv                   # CALCULATE_METRICS
    ├── per_image_metrics_by_grouping.csv     # CREATE_GROUPING_METRICS
    ├── summary_metrics_by_grouping.csv       # CREATE_GROUPING_METRICS
    ├── bar_charts/                           # MAKE_BAR_CHARTS
    │   ├── fg_iou.png
    │   ├── fg_dice.png
    │   ├── precision.png
    │   ├── recall.png
    │   ├── mean_iou.png
    │   ├── mean_dice.png
    │   └── accuracy.png
    ├── mask_grids/                           # MAKE_MASK_GRIDS
    │   └── <image_id>_masks.png
    └── overlay_grids/                        # MAKE_OVERLAY_GRIDS
        └── <image_id>_overlays.png

``per_image_metrics.csv`` contains one row per matched image/model pair, while
``summary_metrics.csv`` contains model-level averages. The two grouping CSVs
contain the corresponding detailed and averaged metrics after joining images
to categories from ``balanced_groupings_test_images.json``. Bar charts compare
model-level averages. Mask grids show ground truth beside each prediction, and
overlay grids draw those masks over the source photograph.

Existing files with the same names are replaced. Metric CSVs use image-level
macro averaging: metrics are computed independently for each image, then the
per-image values are averaged per model or grouping. This gives each image
equal weight regardless of its dimensions.
"""

from pathlib import Path
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAccuracy,
)

# ---------------------------------------------------------------------------
# Input and output configuration
# ---------------------------------------------------------------------------

# Dataset root containing ``gt_masks``, ``images``, and ``predict_masks``.
BASE_DIR = Path(
    "data/nav_seg_model_accuracy_comp_paper/data/test_dataset/"
    "images_masks_grouped/all_images_masks_in_grouping"
)

# Ground-truth masks must be PNG files directly inside this directory.
GT_DIR = BASE_DIR / "gt_masks"
# Source photographs are optional unless overlay grids are requested. Supported
# extensions are .jpg, .jpeg, and .png.
IMAGE_DIR = BASE_DIR / "images"
# Each immediate child directory represents one model and contains PNG masks.
PRED_DIR = BASE_DIR / "predict_masks"
# All generated CSV and image artifacts are placed under this directory.
OUT_DIR = BASE_DIR / "mask_comparison_metrics"
# JSON used to associate image IDs with categories for grouped summaries.
GROUPING_JSON = (
    BASE_DIR.parent.parent / "balanced_groupings_test_images.json"
)

# ---------------------------------------------------------------------------
# Pipeline switches
# ---------------------------------------------------------------------------

# Pipeline options
CALCULATE_METRICS = False 
CREATE_GROUPING_METRICS = False 
MAKE_MASK_GRIDS = False  
MAKE_OVERLAY_GRIDS = False  
MAKE_BAR_CHARTS = True

# Plot settings
PLOT_MAX_SIDE = 700
PLOT_DPI = 80
NEON_GREEN = np.array([57, 255, 20], dtype=np.float32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def load_mask(path):
    """Load an image mask as a two-dimensional boolean tensor.

    Pillow converts the image to a single grayscale channel. Any nonzero pixel
    is then treated as foreground.

    Args:
        path: Path to the mask image.

    Returns:
        A boolean ``torch.Tensor`` where ``True`` denotes foreground.
    """
    mask = np.array(Image.open(path).convert("L"))
    return torch.from_numpy(mask > 0)


def compute_metrics(pred, gt):
    """Compute image-level metrics for a binary prediction and ground truth.

    Uses TorchMetrics `MetricCollection` to compute per-image metrics.
    """
    pred_bool = pred.bool()
    gt_bool = gt.bool()

    # Flatten for torchmetrics
    preds_flat = pred_bool.view(-1)
    targets_flat = gt_bool.view(-1)

    # --- Binary / thresholded metrics (use predictions after thresholding) ---
    bin_metrics = MetricCollection(
        {
            "fg_iou": BinaryJaccardIndex(),
            "fg_dice": BinaryF1Score(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "accuracy": BinaryAccuracy(),
        }
    )

    results = {
        "fg_iou": float("nan"),
        "fg_dice": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "mean_iou": float("nan"),
        "mean_dice": float("nan"),
        "accuracy": float("nan"),
    }

    # Compute foreground metrics via MetricCollection
    try:
        bin_out = bin_metrics(preds_flat.long(), targets_flat.long())
        for k, v in bin_out.items():
            results[k] = v.item()
    except Exception:
        # Fall back to NaN for any metric failure
        pass

    # Compute background metrics by applying jaccard/dice to inverted masks
    try:
        jaccard = BinaryJaccardIndex()
        f1 = BinaryF1Score()
        bg_iou = jaccard((~preds_flat).long(), (~targets_flat).long()).item()
        bg_dice = f1((~preds_flat).long(), (~targets_flat).long()).item()
        if not math.isnan(results["fg_iou"]):
            results["mean_iou"] = float(np.nanmean([bg_iou, results["fg_iou"]]))
        else:
            results["mean_iou"] = bg_iou
        if not math.isnan(results["fg_dice"]):
            results["mean_dice"] = float(np.nanmean([bg_dice, results["fg_dice"]]))
        else:
            results["mean_dice"] = bg_dice
    except Exception:
        pass

    return results


def mask_to_numpy(mask):
    """Copy a mask tensor to CPU and return it as a boolean NumPy array."""
    return mask.cpu().numpy().astype(bool)


def resize_for_plot(array, max_size=None, is_mask=False):
    """Downsample an array for plotting while preserving its aspect ratio.

    Arrays already within ``max_size`` are returned unchanged. Masks use nearest
    neighbor resampling so class labels remain discrete; photographs use
    bilinear resampling. This function affects visualizations only.

    Args:
        array: Image-like NumPy array with height and width as its first axes.
        max_size: Maximum permitted length of either side. Defaults to
            :data:`PLOT_MAX_SIDE`.
        is_mask: Whether to use label-preserving nearest-neighbor interpolation.

    Returns:
        The original array or a resized NumPy array.
    """
    if max_size is None:
        max_size = PLOT_MAX_SIDE

    height, width = array.shape[:2]
    scale = min(max_size / max(height, width), 1)
    if scale == 1:
        return array

    new_size = (int(width * scale), int(height * scale))
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return np.array(Image.fromarray(array).resize(new_size, resample=resample))


def load_photo(image_id):
    """Load an RGB source photograph matching an image ID.

    Extensions are checked in ``.jpg``, ``.jpeg``, then ``.png`` order.

    Returns:
        An ``H x W x 3`` uint8 NumPy array, or ``None`` if no photo exists.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        path = IMAGE_DIR / f"{image_id}{ext}"
        if path.exists():
            return np.array(Image.open(path).convert("RGB"))
    return None


def make_overlay(photo, mask):
    """Blend foreground mask pixels over a photograph in neon green.

    The mask is first resized according to the plotting limit. The photograph
    is then resized to exactly match it. If no photograph is available, a black
    background is used so overlay generation can still proceed.

    Returns:
        An RGB uint8 NumPy array ready for plotting.
    """
    mask_np = resize_for_plot(mask_to_numpy(mask).astype("uint8"), is_mask=True).astype(bool)

    if photo is None:
        photo = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    else:
        photo = np.array(
            Image.fromarray(photo).resize(
                (mask_np.shape[1], mask_np.shape[0]),
                resample=Image.Resampling.BILINEAR,
            )
        )

    overlay = photo.astype(np.float32)
    overlay[mask_np] = (overlay[mask_np] * 0.35) + (NEON_GREEN * 0.65)
    return overlay.astype(np.uint8)


def save_mask_grid(image_id, gt, pred_masks, image_metrics):
    """Save a grid containing ground truth and all supplied predicted masks.

    Args:
        image_id: Filename stem used for the output filename and metric lookup.
        gt: Ground-truth boolean tensor.
        pred_masks: Mapping of model name to predicted boolean tensor.
        image_metrics: Mapping of ``(image_id, model_name)`` to foreground IoU.
    """
    path = OUT_DIR / "mask_grids" / f"{image_id}_masks.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    panels = [("GT", gt)] + list(pred_masks.items())
    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for ax, (title, mask) in zip(axes, panels):
        mask_np = resize_for_plot(mask_to_numpy(mask).astype("uint8") * 255, is_mask=True)
        ax.imshow(mask_np, cmap="gray")
        if title == "GT":
            ax.set_title("GT\nIoU: 1.000", fontsize=9)
        else:
            metric = image_metrics.get((image_id, title))
            metric_text = f"{metric:.3f}" if pd.notna(metric) else "NA"
            ax.set_title(f"{title}\nIoU: {metric_text}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def save_overlay_grid(image_id, gt, pred_masks, image_metrics):
    """Save ground-truth and prediction overlays for one image.

    Missing source photographs are represented with a black background. Panel
    titles show foreground IoU where a computed value is available.
    """
    path = OUT_DIR / "overlay_grids" / f"{image_id}_overlays.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    photo = load_photo(image_id)
    panels = [("GT", gt)] + list(pred_masks.items())
    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for ax, (title, mask) in zip(axes, panels):
        overlay = make_overlay(photo, mask)
        ax.imshow(overlay)
        if title == "GT":
            ax.set_title("GT\nIoU: 1.000", fontsize=9)
        else:
            metric = image_metrics.get((image_id, title))
            metric_text = f"{metric:.3f}" if pd.notna(metric) else "NA"
            ax.set_title(f"{title}\nIoU: {metric_text}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def save_metric_bar_charts(summary_df):
    """Save one model-comparison bar chart for each summary metric.

    ``summary_df`` must contain a ``model`` column and every metric listed in
    this function. Values are expected to be in the range 0 through 1.
    """
    chart_dir = OUT_DIR / "bar_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "fg_iou",
        "fg_dice",
        "precision",
        "recall",
        "mean_iou",
        "mean_dice",
        "accuracy",
    ]
    for metric in metrics:
        logging.info("Creating bar chart for %s", metric)
        plot_df = summary_df.dropna(subset=[metric]).sort_values(metric, ascending=False)
        if plot_df.empty:
            logging.info("Skipping bar chart for %s because it has no values", metric)
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(plot_df["model"], plot_df[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=75, labelsize=8)
        for index, value in enumerate(plot_df[metric]):
            ax.text(index, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(chart_dir / f"{metric}.png", dpi=PLOT_DPI)
        plt.close(fig)


def load_grouping_table():
    """Convert the grouping JSON into a long-form image membership table.

    The accepted top-level format is either the grouping mapping itself or an
    object containing that mapping under ``"test_grouping"``. Its nested shape
    is expected to be::

        {
            "grouping_type": {
                "grouping_name": ["path/to/image_001.jpg", "..."]
            }
        }

    Filename stems become ``image_id`` values so they can be joined to the
    per-image metric table. Duplicate memberships are removed.
    """
    logging.info("Loading grouping JSON: %s", GROUPING_JSON)
    with GROUPING_JSON.open() as f:
        data = json.load(f)

    grouping_data = data.get("test_grouping", data)
    rows = []
    for grouping_type, groups in grouping_data.items():
        for grouping_name, filenames in groups.items():
            for filename in filenames:
                rows.append({
                    "image_id": Path(filename).stem,
                    "grouping_type": grouping_type,
                    "grouping": grouping_name,
                })

    grouping_df = pd.DataFrame(rows).drop_duplicates()
    logging.info("Loaded %d image grouping rows", len(grouping_df))
    return grouping_df


def save_grouping_metrics(per_image_df):
    """Join metrics to image groupings and save detailed and summary CSVs.

    Only images present in both the metric table and grouping JSON are retained
    because the merge is an inner join. Images may appear in multiple grouping
    types and therefore contribute to multiple summary rows.

    Args:
        per_image_df: DataFrame produced by the metric calculation stage.
    """
    if per_image_df.empty:
        logging.warning("Cannot create grouping metrics because per-image metrics are empty")
        return

    grouping_df = load_grouping_table()
    grouped_per_image_df = per_image_df.merge(grouping_df, on="image_id", how="inner")

    metrics = [
        "fg_iou",
        "fg_dice",
        "precision",
        "recall",
        "mean_iou",
        "mean_dice",
        "accuracy",
    ]
    grouped_summary_df = (
        grouped_per_image_df
        .groupby(["model", "grouping_type", "grouping"], as_index=False)
        .agg({
            "image_id": "count",
            **{metric: "mean" for metric in metrics},
        })
        .rename(columns={"image_id": "num_compared"})
        .sort_values(["grouping_type", "grouping", "fg_iou"], ascending=[True, True, False])
    )

    grouped_per_image_path = OUT_DIR / "per_image_metrics_by_grouping.csv"
    grouped_summary_path = OUT_DIR / "summary_metrics_by_grouping.csv"
    grouped_per_image_df.to_csv(grouped_per_image_path, index=False)
    grouped_summary_df.to_csv(grouped_summary_path, index=False)

    logging.info("Wrote grouped per-image metrics: %s", grouped_per_image_path)
    logging.info("Wrote grouped summary metrics: %s", grouped_summary_path)


def main():
    """Run the configured metric, grouping, chart, and visualization stages.

    The calculation stage first matches masks by filename stem within each
    model. Optional visualization stages independently re-index predictions so
    they can also run when metrics are loaded from existing CSV files.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Starting mask comparison")
    logging.info("GT directory: %s", GT_DIR)
    logging.info("Image directory: %s", IMAGE_DIR)
    logging.info("Prediction directory: %s", PRED_DIR)
    logging.info("Output directory: %s", OUT_DIR)
    logging.info("Indexing ground-truth masks")
    gt_masks = {p.stem: p for p in GT_DIR.glob("*.png")}
    logging.info("Indexing model prediction folders")
    model_dirs = sorted([p for p in PRED_DIR.iterdir() if p.is_dir()])
    per_image_rows = []
    skipped = 0

    logging.info("Found %d ground-truth masks", len(gt_masks))
    logging.info("Found %d model folders", len(model_dirs))

    if CALCULATE_METRICS:
        for model_dir in model_dirs:
            model_name = model_dir.name
            logging.info("Indexing predicted masks for model: %s", model_name)
            pred_masks = {p.stem: p for p in (model_dir / "masks").glob("*.png")}
            common_images = sorted(set(gt_masks) & set(pred_masks))

            logging.info(
                "Computing metrics for model: %s (%d matched images)",
                model_name,
                len(common_images),
            )

            for image_id in tqdm(common_images, desc=model_name, unit="image"):
                gt = load_mask(gt_masks[image_id])
                pred = load_mask(pred_masks[image_id])

                if pred.shape != gt.shape:
                    logging.warning(
                        "Skipping %s for %s: shape mismatch pred=%s gt=%s",
                        image_id,
                        model_name,
                        tuple(pred.shape),
                        tuple(gt.shape),
                    )
                    skipped += 1
                    continue

                metrics = compute_metrics(pred, gt)

                per_image_rows.append({
                    "model": model_name,
                    "image_id": image_id,
                    **metrics,
                })

            logging.info("Finished model: %s", model_name)

        logging.info("Building per-image metrics table")
        per_image_df = pd.DataFrame(per_image_rows)
        logging.info("Computed metrics for %d image/model pairs", len(per_image_df))

        logging.info("Building summary metrics table")
        summary_df = (
            per_image_df
            .groupby("model", as_index=False)
            .agg({
                "image_id": "count",
                "fg_iou": "mean",
                "fg_dice": "mean",
                "precision": "mean",
                "recall": "mean",
                "mean_iou": "mean",
                "mean_dice": "mean",
                "accuracy": "mean",
            })
            .rename(columns={"image_id": "num_compared"})
            .sort_values("fg_iou", ascending=False)
        )

        logging.info("Writing metric CSV files")
        per_image_df.to_csv(OUT_DIR / "per_image_metrics.csv", index=False)
        summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    else:
        logging.info("Skipping metric calculation because CALCULATE_METRICS=False")
        per_image_path = OUT_DIR / "per_image_metrics.csv"
        if per_image_path.exists():
            logging.info("Reading existing per-image metrics: %s", per_image_path)
            per_image_df = pd.read_csv(per_image_path)
        else:
            logging.warning("No existing per-image metrics found at: %s", per_image_path)
            per_image_df = pd.DataFrame()

    if MAKE_BAR_CHARTS:
        summary_path = OUT_DIR / "summary_metrics.csv"
        if CALCULATE_METRICS:
            logging.info("Creating metric bar charts from newly calculated metrics")
        elif summary_path.exists():
            logging.info("Reading existing summary metrics for bar charts: %s", summary_path)
            summary_df = pd.read_csv(summary_path)
        else:
            logging.warning("Cannot make bar charts; missing summary metrics: %s", summary_path)
            summary_df = pd.DataFrame()

        if not summary_df.empty:
            logging.info("Creating metric bar charts")
            save_metric_bar_charts(summary_df)

    if CREATE_GROUPING_METRICS:
        logging.info("Creating grouped metric CSV files")
        save_grouping_metrics(per_image_df)

    saved_images = 0
    if MAKE_MASK_GRIDS or MAKE_OVERLAY_GRIDS:
        logging.info("Selecting images for mask and overlay figures")
        image_ids_to_save = sorted(gt_masks)

        logging.info("Indexing predictions again for figure generation")
        pred_masks_by_model = {
            model_dir.name: {p.stem: p for p in (model_dir / "masks").glob("*.png")}
            for model_dir in model_dirs
        }
        image_metrics = {}
        if not per_image_df.empty:
            image_metrics = {
                (row["image_id"], row["model"]): row["fg_iou"]
                for _, row in per_image_df.iterrows()
            }

        for image_id in tqdm(image_ids_to_save, desc="Saving comparison figures", unit="image"):
            gt = load_mask(gt_masks[image_id])
            pred_masks_for_plot = {}

            for model_name, pred_masks in pred_masks_by_model.items():
                if image_id not in pred_masks:
                    continue

                pred = load_mask(pred_masks[image_id])
                if pred.shape == gt.shape:
                    pred_masks_for_plot[model_name] = pred

            if pred_masks_for_plot:
                if MAKE_MASK_GRIDS:
                    save_mask_grid(image_id, gt, pred_masks_for_plot, image_metrics)
                if MAKE_OVERLAY_GRIDS:
                    save_overlay_grid(image_id, gt, pred_masks_for_plot, image_metrics)
                saved_images += 1

    logging.info("Skipped %d image/model pairs", skipped)
    if CALCULATE_METRICS:
        logging.info("Wrote: %s", OUT_DIR / "per_image_metrics.csv")
        logging.info("Wrote: %s", OUT_DIR / "summary_metrics.csv")
    if MAKE_BAR_CHARTS:
        logging.info("Wrote bar charts under: %s", OUT_DIR / "bar_charts")
    if CREATE_GROUPING_METRICS:
        logging.info("Wrote grouping metrics under: %s", OUT_DIR)
    if MAKE_MASK_GRIDS:
        logging.info("Saved mask grid figures under: %s", OUT_DIR / "mask_grids")
    if MAKE_OVERLAY_GRIDS:
        logging.info("Saved overlay grid figures under: %s", OUT_DIR / "overlay_grids")
    if MAKE_MASK_GRIDS or MAKE_OVERLAY_GRIDS:
        logging.info("Created figures for %d images", saved_images)


if __name__ == "__main__":
    main()
