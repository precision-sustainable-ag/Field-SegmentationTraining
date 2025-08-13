"""
UNet Segmentation Inference Script
==================================
This script performs semantic segmentation on crop images using a pre-trained UNet model.
It reads images and bounding box metadata, crops the region of interest, predicts a segmentation mask, and saves the results.
"""
import json
from typing import Dict, Optional, Tuple
import cv2
import torch
import logging
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime
import segmentation_models_pytorch as smp
from torchvision import transforms

# Logging configuration
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetInference:

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg.paths.base_dir
            cfg.paths.project_maskgen_dir
            cfg.paths.input_csv          # CSV to read
            cfg.paths.output_csv         # CSV to write (optional; defaults to input)
            cfg.paths.unet_segmentation_model
        """
        log.info(f"Initializing UNetInference at {datetime.now()}")
        self.cfg = cfg

        self._setup_paths()
        self._load_dataframe()
        self._prepare_dataframe_columns()
        self._load_model()
        self._setup_transforms()

    def _setup_paths(self) -> None:
        """Initialize all path-related attributes and ensure output dirs exist."""
        self.repo_root = Path(self.cfg.paths.base_dir)
        self.maskgen_dir = Path(self.cfg.paths.project_maskgen_dir)
        self.developed_images_dir = self.maskgen_dir / "developed-images"
        self.cutout_dir = self.maskgen_dir / "cutouts"
        self.cutout_dir.mkdir(parents=True, exist_ok=True)
        self.initial_mask_dir = self.maskgen_dir / "initial_masks"
        self.initial_mask_dir.mkdir(parents=True, exist_ok=True)

        self.input_csv = Path(self.cfg.paths.project_temp_db)
        self.output_csv = self.input_csv

        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")
    
    def _load_dataframe(self) -> None:
        """Load the input CSV into a DataFrame."""
        self.df = pd.read_csv(self.input_csv)
    
    def _prepare_dataframe_columns(self) -> None:
        """Ensure required output columns exist in the DataFrame."""
        for col in ("initial_mask_path", "seg_note"):
            if col not in self.df.columns:
                self.df[col] = pd.Series([None] * len(self.df), dtype="object")
            else:
                # If the column already exists, convert its type to object
                self.df[col] = self.df[col].astype("object")
    
    def _load_model(self) -> None:
        """Load the UNet segmentation model and weights."""
        self.trained_model_path = Path(self.cfg.paths.unet_segmentation_model)
        self.seg_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(DEVICE)

        state = torch.load(
            self.trained_model_path,
            map_location=DEVICE,
            weights_only=True
        )
        self.seg_model.load_state_dict(state)
        self.seg_model.eval()

    def _setup_transforms(self) -> None:
        """Prepare torchvision transforms for inference."""
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _resolve_image_path(self, row: pd.Series) -> Optional[Path]:
        """
        Prefer 'local_developed_image_path'. Fallback to project/developed-images/<stem>.<ext>.
        """
        # 1) local_developed_image_path
        local_rel = row.get("local_developed_image_path")
        if isinstance(local_rel, str) and local_rel.strip():
            p = Path(local_rel)
            if not p.is_absolute():
                p = (self.repo_root / p).resolve()
            if p.exists():
                return p

        # 2) derive from stem + extension under project dir
        stem = row.get("stem")
        ext = (row.get("extension") or "jpg").lower()
        if isinstance(stem, str) and stem:
            cand = (self.developed_images_dir / f"{stem}.{ext}").resolve()
            if cand.exists():
                return cand

        # 3) last resort: try image_id
        image_id = row.get("image_id")
        if isinstance(image_id, str) and image_id:
            cand = (self.developed_images_dir / image_id).with_suffix(f".{ext}")
            if cand.exists():
                return cand.resolve()

        return None
    
    @staticmethod
    def _parse_bbox_xywh(row: pd.Series) -> Optional[Tuple[int, int, int, int]]:
        """
        Expects 'bbox_xywh' as a JSON array string "[x, y, w, h]" (float or int).
        Returns (x, y, w, h) as ints, or None if missing/invalid.
        """
        val = row.get("bbox_xywh")
        if not isinstance(val, str) or not val.strip():
            return None
        try:
            arr = json.loads(val)
            if not (isinstance(arr, list) and len(arr) == 4):
                return None
            x, y, w, h = [int(round(float(v))) for v in arr]
            if w <= 0 or h <= 0:
                return None
            return x, y, w, h
        except Exception:
            return None

    @staticmethod
    def _bbox_to_minmax(bbox_xywh: Tuple[int, int, int, int]) -> Dict[str, int]:
        x, y, w, h = bbox_xywh
        return {"x_min": x, "x_max": x + w, "y_min": y, "y_max": y + h}

    
    def _predict_mask(self, cropped_rgb: np.ndarray) -> np.ndarray:
        """
        Returns a binary mask (0/1) the same HxW as cropped_rgb.
        """
        pil_image = Image.fromarray(cropped_rgb)
        image_tensor = self.transform(pil_image).float().to(DEVICE).unsqueeze(0)

        log.debug(f"Predicting mask for tensor: {tuple(image_tensor.shape)}")
        # New version with torch.no_grad()
        with torch.no_grad():
            logits = self.seg_model(image_tensor)
            prob = torch.sigmoid(logits).squeeze(0).cpu().detach().permute(1,2,0)
            pred_mask = (prob > 0.5).float().numpy().squeeze(-1)
        
        log.debug(f"Logits shape: {logits.shape}, Probability shape: {prob.shape}")
        log.debug(f"Predicted mask shape: {pred_mask.shape}")

        return pred_mask

    def _predict_mask_in_tiles(self, image_rgb: np.ndarray, overlap_pixels=250, max_tile_size=4500) -> np.ndarray:
        """
        Tile-based inference for very large crops.
        """
        H, W = image_rgb.shape[:2]
        step_h = min(int(np.ceil(H / 2)), max_tile_size - overlap_pixels)
        step_w = min(int(np.ceil(W / 2)), max_tile_size - overlap_pixels)
        tile_h, tile_w = step_h + overlap_pixels, step_w + overlap_pixels

        out = np.zeros((H, W), dtype=np.uint8)
        for y in range(0, H, step_h):
            for x in range(0, W, step_w):
                y2, x2 = min(y + tile_h, H), min(x + tile_w, W)
                tile = image_rgb[y:y2, x:x2]
                tile_mask = self._predict_mask(tile)
                # Combine (OR)
                out[y:y2, x:x2] = np.maximum(out[y:y2, x:x2], tile_mask)
                # out[y:y2, x:x2] = np.maximum(out[y:y2, x:x2], tile_mask.squeeze())
        return out
    

    def _embed_mask(self, pred_mask_crop: np.ndarray, bbox: Dict[str, int], full_hw: Tuple[int, int]) -> np.ndarray:
        """
        Embed a crop-sized mask back into full-size image space.
        """
        H, W = full_hw
        x_min, x_max, y_min, y_max = bbox["x_min"], bbox["x_max"], bbox["y_min"], bbox["y_max"]
        crop_h, crop_w = y_max - y_min, x_max - x_min

        if pred_mask_crop.shape[:2] != (crop_h, crop_w):
            pred_mask_crop = cv2.resize(pred_mask_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = pred_mask_crop
        return full_mask


    def process_row(self, idx: int, row: pd.Series) -> None:
        img_path = self._resolve_image_path(row)
        if img_path is None:
            self.df.loc[idx, "seg_note"] = "Image not found"
            return

        bbox_xywh = self._parse_bbox_xywh(row)
        if bbox_xywh is None:
            self.df.loc[idx, "seg_note"] = "Missing/invalid bbox_xywh"
            return

        bx = self._bbox_to_minmax(bbox_xywh)

        # Load and crop (OpenCV loads BGR; convert to RGB for model)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            self.df.loc[idx, "seg_note"] = "Failed to read image"
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Safety clamp bbox to image bounds
        H, W = rgb.shape[:2]
        bx["x_min"] = max(0, min(bx["x_min"], W))
        bx["x_max"] = max(0, min(bx["x_max"], W))
        bx["y_min"] = max(0, min(bx["y_min"], H))
        bx["y_max"] = max(0, min(bx["y_max"], H))
        if bx["x_max"] <= bx["x_min"] or bx["y_max"] <= bx["y_min"]:
            self.df.loc[idx, "seg_note"] = "Empty crop after clamp"
            return

        crop = rgb[bx["y_min"]:bx["y_max"], bx["x_min"]:bx["x_max"]]

        # Predict (tile if very large)
        if crop.shape[0] > 4000 or crop.shape[1] > 4000:
            pred_crop = self._predict_mask_in_tiles(crop)
        else:
            pred_crop = self._predict_mask(crop)

        # Embed back to full image (binary 0/1)
        full_mask = self._embed_mask(pred_crop, bx, (H, W))

        # Build outputs & save
        stem = Path(img_path).stem
        cutout_name = f"{stem}_0.png"
        final_mask_name = f"{stem}_0_mask.png"
        cropout_name = f"{stem}_0.jpg"

        # Save crop image (for review), mask, and cutout
        # (Note: cutout here simply zeroes background using the crop mask)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.cutout_dir / cropout_name), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(self.cutout_dir / final_mask_name), (pred_crop * 255).astype(np.uint8))

        # Create a 3-channel masked crop as PNG
        crop_mask_3c = np.repeat(pred_crop[:, :, None], 3, axis=2).astype(np.uint8)
        cutout_bgr = np.where(crop_mask_3c == 1, crop_bgr, 0)
        cv2.imwrite(str(self.cutout_dir / cutout_name), cutout_bgr)

        # Save the initial mask as a PNG
        full_mask_name = f"{stem}_mask.png"
        mask_full_rel = (self.initial_mask_dir / full_mask_name).relative_to(self.repo_root)
        cv2.imwrite(str(self.initial_mask_dir / full_mask_name), (full_mask * 255).astype(np.uint8))


        # Record repo-relative mask path
        mask_rel = (self.cutout_dir / final_mask_name)
        try:
            mask_rel = mask_rel.relative_to(self.repo_root)
        except ValueError:
            # If cutout_dir isn't inside repo_root, fall back to absolute path
            mask_rel = mask_rel

        # Update CSV row
        self.df.loc[idx, "initial_mask_path"] = str(mask_full_rel)
        self.df.loc[idx, "initial_cutout_mask_path"] = str(mask_rel)
        self.df.loc[idx, "seg_note"] = pd.NA

    def run(self) -> None:
        updated, skipped = 0, 0
        for idx, row in self.df.iterrows():
            before = self.df.loc[idx, ["initial_mask_path", "seg_note"]].copy()
            self.process_row(idx, row)
            after = self.df.loc[idx, ["initial_mask_path", "seg_note"]]
            if not after.equals(before):
                updated += 1
            else:
                skipped += 1
        log.info(f"Segmentation updated rows: {updated}; unchanged/skipped: {skipped}")
        self.df.to_csv(self.output_csv, index=False)
        log.info(f"Saved updated CSV to: {self.output_csv}")


def main(cfg: DictConfig) -> None:
    """
    Entry point for running the UNet segmentation inference pipeline.

    Args:
        cfg (DictConfig): Configuration with paths to input image directory and trained model.
    """
    log.info(f"Starting UNet segmentation inference.")
    unet_inference = UNetInference(cfg)
    unet_inference.run()
    log.info(f"UNet segmentation inference complete.")