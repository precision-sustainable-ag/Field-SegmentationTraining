import json
import logging
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from omegaconf import DictConfig
from typing import Optional, Dict

# Configure logging
log = logging.getLogger(__name__)

class WeedDetector:
    """
    A class for detecting weeds in images using a YOLO model.

    This class loads a pretrained YOLO model and provides functionality to detect
    weeds in images. It returns the most confident detection if multiple are found.
    """

    def __init__(self, yolo_model_path: Path) -> None:
        """
        Initializes the WeedDetector instance with a trained YOLO model.

        Args:
            yolo_model_path (Path): Path to the YOLO model weights.
        """
        self.model = YOLO(yolo_model_path)
        self.missing_detection_notes = []  # List to store notes if detections are missing or multiple

    def detect_weeds(self, image_path: Path) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Detects weeds in a given image using the YOLO model.

        If multiple detections are found, the one with the highest confidence score is used.
        If no detections are found, the function returns None.

        Args:
            image_path (Path): Path to the input image.

        Returns:
            Optional[Dict[str, Union[list, float]]]: A dictionary with:
                - "bbox": [x_min, y_min, width, height] of the detected weed
                - "det_pred_conf": Confidence score of the detection (float, rounded to 6 decimals)
                Returns None if no detection is found.
        """
        log.info(f"Detecting weeds in image: {image_path}")
        results = self.model(image_path)

        if not results or not results[0].boxes.xyxy.tolist():
            log.warning("No detection found.")
            self.missing_detection_notes.append("No detection found.")
            return None

        boxes = results[0].boxes.xyxy.tolist()
        confs = results[0].boxes.conf.tolist()

        if len(boxes) > 1:
            log.warning("Multiple detections found. Selecting the one with highest confidence.")
            self.missing_detection_notes.append("Multiple detections. Selected highest confidence.")
            max_conf_idx = confs.index(max(confs))
        else:
            max_conf_idx = 0

        x_min, y_min, x_max, y_max = map(round, boxes[max_conf_idx])
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        det_pred_conf = round(confs[max_conf_idx], 6)

        return {
            "bbox": bbox,
            "det_pred_conf": det_pred_conf
        }

class ProcessDetections:
    """
    A class for batch processing of weed detection in a directory of images.

    Handles reading images, running detection using WeedDetector, and saving
    detection metadata as JSON files.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the ProcessDetections class.

        Args:
            cfg (DictConfig): Hydra/OmegaConf configuration with required paths.
                Required keys: cfg.paths.project_maskgen_dir, cfg.paths.yolo_weed_detection_model
        """
        self.repo_root = Path(cfg.paths.base_dir)
        self.mask_gen_dir = Path(cfg.paths.project_maskgen_dir)
        self.input_csv = self.mask_gen_dir / "temp_db.csv"
        self.weed_detector = WeedDetector(Path(cfg.paths.yolo_weed_detection_model))

        self.results = []

    def _resolve_image_path(self, row: pd.Series) -> Optional[Path]:
        """
        Prefer 'local_developed_image_path' if present; otherwise fall back to mask_gen_dir/developed-images/<stem>.jpg
        """
        # 1) local_developed_image_path
        local_rel = row.get("local_developed_image_path")
        if isinstance(local_rel, str) and local_rel.strip():
            # Handle paths that are already absolute or repo-relative
            p = Path(local_rel)
            if not p.is_absolute():
                p = (self.repo_root / p).resolve()
            if p.exists():
                return p

        # 2) derived by stem + extension in the project dir
        stem = row.get("stem") or Path(str(row.get("image_id", ""))).stem
        ext = (row.get("extension") or "jpg").lower()
        candidate = (self.mask_gen_dir / "developed-images" / f"{stem}.{ext}").resolve()
        return candidate if candidate.exists() else None

    def process_temp_db(self) -> None:
        log.info(f"Loading CSV: {self.input_csv}")
        df = pd.read_csv(self.input_csv)

        # Ensure columns exist
        for col in ("bbox_xywh", "det_pred_conf", "detection_note"):
            if col not in df.columns:
                df[col] = pd.NA

        updated, missing_files = 0, 0
        for idx, row in df.iterrows():
            img_path = self._resolve_image_path(row)
            if img_path is None:
                df.loc[idx, "detection_note"] = "Image not found"
                missing_files += 1
                continue

            det = self.weed_detector.detect_weeds(img_path)
            if det is None:
                df.loc[idx, ["bbox_xywh", "det_pred_conf", "detection_note"]] = [pd.NA, pd.NA, "No detection"]
                continue

            # Store bbox as JSON string to be CSV-safe and easy to parse later
            df.loc[idx, "bbox_xywh"] = json.dumps(det["bbox"])
            df.loc[idx, "det_pred_conf"] = det["det_pred_conf"]
            df.loc[idx, "detection_note"] = pd.NA
            updated += 1

        log.info(f"Detections updated: {updated}; Missing images: {missing_files}")
        df.to_csv(self.input_csv, index=False)
        log.info(f"Saved updated CSV to: {self.input_csv}")


def main(cfg: DictConfig) -> None:
    """
    Entry point for running the weed detection process using configuration settings.

    Args:
        cfg (DictConfig): Configuration object containing model path and image directories.
    """
    log.info("Starting weed detection process...")
    ProcessDetections(cfg).process_temp_db()
    log.info("Weed detection process completed.")