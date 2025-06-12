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
    A class for detecting weeds in images using YOLOv5.

    This class initializes a YOLOv5 model and provides methods for detecting weeds
    within images, extracting bounding box coordinates, and associated confidence scores.
    It also handles cases of no detection or multiple detections by selecting the
    detection with the highest confidence.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the WeedDetector class.

        Args:
            model_path (str): Path to the YOLOv5 model weights (e.g., 'path/to/best.pt').
        """
        self.model = YOLO(model_path)
        self.missing_detection_notes = [] # list to store missing detection notes
    
    def detect_weeds(self, image_path: Path) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Detects target weeds in the given image.

        This method takes an image path, runs the YOLOv5 model to detect weeds,
        and processes the results. If multiple detections are found, it selects
        the one with the highest confidence. If no detections are found, it logs a warning.

        Args:
            image_path (Path): The file path to the image to be processed.

        Returns:
            Optional[Dict[str, Dict[str, int]]]: A dictionary containing detection results
            if successful. The dictionary includes:
                - "bbox" (list): A list of four integers [x_min, y_min, width, height]
                  representing the bounding box of the detected weed.
                - "det_pred_conf" (float): The confidence score of the detection,
                  rounded to 6 decimal places.
            Returns None if no detection is found.
        """
        log.info("Starting weed detection.")
        results = self.model(image_path)

        if not results or not results[0].boxes.xyxy.tolist():
            log.warning("No detection found.")
            self.missing_detection_notes.append("No detection found.")
            bbox = None
            det_pred_conf = None
        
        else:
            if len(results[0].boxes.xyxy.tolist()) > 1:
                log.warning("More than one detection found. Using the one with the highest confidence.")
                self.missing_detection_notes.append("More than one detection found. Using the one with the highest confidence.")
                # choose the detection with the highest confidence
                confidences = [x.conf for x in results[0].boxes]
                max_conf_idx = confidences.index(max(confidences))
                bbox = results[0].boxes.xyxy.tolist()[max_conf_idx]
                # confidence of the detection
                det_pred_conf = round(results[0].boxes.conf[max_conf_idx].item(), 6)
            else:
                # Extract the detection confidence score
                det_pred_conf = round(results[0].boxes.conf.item(), 6)
                # Extract the bounding box coordinates
                bbox = results[0].boxes.xyxy.tolist()[0]
            
            x_min, y_min, x_max, y_max = map(round, bbox)
            bbox_height = y_max - y_min
            bbox_width = x_max - x_min
            bbox = [x_min, y_min, bbox_width, bbox_height]
        return {
        "bbox": bbox,
        "det_pred_conf": det_pred_conf
        }

class ProcessDetections:
    """
    A class to orchestrate the weed detection and metadata extraction process.

    This class manages the overall workflow of detecting weeds in a directory of images,
    saving the detection results as JSON metadata files.
    """

    def __init__(self, yolo_model_path: Path, batch_dir: Path) -> None:
        """
        Initializes the ProcessDetections class.

        Args:
            yolo_model_path (Path): Path to the YOLOv5 model weights.
            batch_dir (Path): Path to the main input directory containing 'developed-images'.
        """
        self.batch_dir = batch_dir
        self.weed_detector = WeedDetector(yolo_model_path)
        self.detection_save_dir = batch_dir / "cutouts" 
        self.detection_save_dir.mkdir(exist_ok=True)

    def process_image(self, image_path: Path) -> None:
        """
        Processes a single image by detecting weeds and saving the detection metadata.

        The detection results (bounding box and confidence) are saved to a JSON file
        within a 'cutouts' subdirectory, named according to the image stem.

        Args:
            image_path (Path): Path to the image file to be processed.
        """
        log.info(f"Processing image: {image_path}")
        # save the detection results in the 'cutouts' directory
        detection_json_path = self.detection_save_dir / f"{image_path.stem}.json"
        
        # Get detection results
        detection_results = self.weed_detector.detect_weeds(image_path)
        # Save detection results to a JSON file
        with open(detection_json_path, "w") as f:
            json.dump(detection_results, f, indent=4)
        log.info(f"Saved detection results to: {detection_json_path}")
    
    def process_dir(self) -> None:
        """
        Processes all images within the specified input directory.

        This method iterates through all JPG images in the 'developed-images' subdirectory
        of the input directory, performs weed detection on each, and saves the results.
        It also creates a 'cutouts' directory if it doesn't already exist to store the JSON output.
        """
        log.info(f"Processing directory: {self.batch_dir}")
        image_dir = self.batch_dir / "developed-images"

        # Loop through the images in the image directory
        image_paths = sorted(list(image_dir.glob("*.jpg")))

        for image_path in image_paths:
            self.weed_detector.missing_detection_notes = []
            self.process_image(image_path)
        log.info(f"Processed {len(image_paths)} images in {image_dir}.")

yolo_model_path = Path("/home/nsingh27/Field-SegmentationTraining/mask_generation/data/field-tools/models/weed_detection/weights/best.pt")
batch_dir = Path("/home/nsingh27/Field-SegmentationTraining/mask_generation/data/image_processing_dir")
processdetections = ProcessDetections(yolo_model_path, batch_dir)
processdetections.process_dir()