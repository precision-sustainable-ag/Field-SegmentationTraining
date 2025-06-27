"""
Weed Detection with YOLO
==========================

This script detects weeds in images using a trained YOLO model. It processes all .jpg images in the
'developed-images' folder and saves detection metadata (bounding box and confidence) as JSON files
in the 'cutouts' directory.
"""

import json
import logging
from pathlib import Path
from ultralytics import YOLO
from omegaconf import DictConfig
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
                Required keys: cfg.paths.mask_gen_dir, cfg.paths.yolo_weed_detection_model
        """
        self.mask_gen_dir = Path(cfg.paths.mask_gen_dir)
        self.weed_detector = WeedDetector(Path(cfg.paths.yolo_weed_detection_model))
        self.detection_save_dir = self.mask_gen_dir / "cutouts"
        self.detection_save_dir.mkdir(exist_ok=True)

    def process_image(self, image_path: Path) -> None:
        """
        Processes a single image by performing weed detection and saving metadata.

        Saves results to a JSON file named after the image stem in the `cutouts` directory.

        Args:
            image_path (Path): Path to the image to be processed.
        """
        log.info(f"Processing image: {image_path.name}")
        detection_results = self.weed_detector.detect_weeds(image_path)

        detection_json_path = self.detection_save_dir / f"{image_path.stem}_0.json"
        with open(detection_json_path, "w") as f:
            json.dump(detection_results, f, indent=4)

        log.info(f"Saved detection results to: {detection_json_path}")

    def process_dir(self) -> None:
        """
        Processes all JPG images in the 'developed-images' folder of the root directory.
        
        Applies weed detection and stores output metadata in JSON format under 'cutouts/'.
        """
        image_dir = self.mask_gen_dir / "developed-images"
        image_paths = sorted(image_dir.glob("*.jpg"))
        log.info(f"Found {len(image_paths)} images in {image_dir}. Starting detection...")

        for image_path in image_paths:
            self.weed_detector.missing_detection_notes = []
            self.process_image(image_path)

        log.info("Completed processing all images.")

def main(cfg: DictConfig) -> None:
    """
    Entry point for running the weed detection process using configuration settings.

    Args:
        cfg (DictConfig): Configuration object containing model path and image directories.
    """
    log.info("Starting weed detection process...")
    detector = ProcessDetections(cfg)
    detector.process_dir()
    log.info("Weed detection process completed.")