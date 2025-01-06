import os
import cv2
import torch
import logging
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from torch.nn import DataParallel
from torchvision import transforms
from src.utils.unet import UNet
from omegaconf import DictConfig

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class UNetInference:
    """
    A class for performing inference using a pre-trained UNet model.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the UNetInference class, load the model, and prepare directories.

        Args:
            cfg (DictConfig): Configuration object containing paths and settings.
        """
        log.info(f"Initializing UNetInference at {datetime.now()}")
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Set up directories
        self.model_save_dir = Path(cfg.paths.model_save_dir)
        self.test_dir = Path(cfg.paths.test_dir)

        self.inference_results_dir = Path(cfg.paths.inference_results_dir)
        self.inference_results_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp_inference_results_dir = self.inference_results_dir / timestamp
        self.timestamp_inference_results_dir.mkdir(parents=True, exist_ok=True)

        self.masks_dir = self.timestamp_inference_results_dir / "masks"
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        self.img_mask_comparison_dir = self.timestamp_inference_results_dir / "img_mask_overlayed"
        self.img_mask_comparison_dir.mkdir(parents=True, exist_ok=True)

        # Find the latest trained model
        self.latest_folder = self._get_latest_folder()
        if not self.latest_folder:
            log.error("No valid folder with a trained model was found.")
            raise FileNotFoundError("No valid folder with a trained model was found.")

        self.trained_model_path = os.path.join(
            self.model_save_dir, self.latest_folder, "project/weights/unet_segmentation.pth"
        )
        log.info(f"Trained UNet model located at: {self.trained_model_path}")

        # Load pretrained weed detection YOLO model
        log.info("Loading YOLO model for target weed detection.")
        self.target_weed_detection_model = YOLO(cfg.paths.yolo_weed_detection_model)

        # Load UNet model
        log.info("Loading UNet model for segmentation.")
        self.seg_model = UNet(in_channels=3, num_classes=1).to(device)
        self.seg_model.load_state_dict(torch.load(self.trained_model_path, map_location=device))
        self.seg_model.eval()
        log.info("UNet model loaded and set to evaluation mode.")

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _get_latest_folder(self) -> str:
        """
        Finds the folder with the latest date in its name.

        Returns:
            str: The name of the folder with the closest date to the current date.
        """
        min_difference = float("inf")
        latest_folder = None

        for folder_name in os.listdir(self.model_save_dir):
            try:
                folder_date = folder_name.split('_')[1]
                folder_date_object = datetime.strptime(folder_date, "%Y-%m-%d")
                difference = abs((folder_date_object - datetime.now()).days)
                if difference < min_difference:
                    min_difference = difference
                    latest_folder = folder_name
            except (IndexError, ValueError):
                log.warning(f"Invalid dated folder name: {folder_name}")
        
        log.info(f"Latest folder identified: {latest_folder}")
        return latest_folder

    def _convert_to_rgb(self, image_path: str) -> np.ndarray:
        """
        Converts the given image to RGB format.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Image as a numpy array in RGB format.
        """
        image = cv2.imread(str(image_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _detect_weeds(self, image_path: str):
        """
        Detects target weed in the given image using a pretrained model.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple or None: Bounding box coordinates (x_min, y_min, x_max, y_max) if detection is successful; otherwise, None.
        """
        log.info("Starting weed detection.")
        image = self._convert_to_rgb(image_path)
        results = self.target_weed_detection_model(image)

        if not results or not results[0].boxes.xyxy.tolist():
            log.warning(f"No detection found in image: {image_path}")
            return None

        bbox = results[0].boxes.xyxy.tolist()[0]
        return tuple(map(int, bbox))

    def _crop_image_bbox(self, image_path: str):
        """
        Crops the image using the detected bounding box.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: Cropped image, bounding box, and original image.
        """
        image_full_size = self._convert_to_rgb(image_path)
        bbox = self._detect_weeds(image_path)
        if bbox is None:
            log.error(f"Could not detect weed in image: {image_path}")
            raise ValueError("Weed detection failed.")

        x_min, y_min, x_max, y_max = bbox
        cropped_image = image_full_size[y_min:y_max, x_min:x_max]

        return cropped_image, bbox, image_full_size

    def infer_single_image(self, image_path: str):
        """
        Performs segmentation mask prediction for a single image and saves the result.

        Args:
            image_path (str): Path to the input image.
        """
        log.info(f"Processing image: {image_path}")
        image_name = Path(image_path).stem
        cropped_image, bbox, image_full_size = self._process_image(image_path)
        cropped_image_shape = cropped_image.shape

        log.info(f"Size of cropped image: {cropped_image_shape}")

        if cropped_image_shape[0] < 4000 or cropped_image_shape[1] < 3000:
            log.info(f"Image size is small enough for direct processing.")
            pred_mask = self._predict_mask(cropped_image)
        else:
            log.info(f"Image size is too big for direct processing. Resizing by 2 and processing.")
            pred_mask = self._predict_mask(cv2.resize(cropped_image, (cropped_image_shape[1] // 2, cropped_image_shape[0] // 2)))

        padded_mask = self._resize_and_pad_mask(pred_mask, bbox, image_full_size.shape[:2])

        self._save_full_size_mask(padded_mask, image_name)
        self._save_overlay_image(padded_mask, image_full_size, image_name)

        log.info(f"Processing completed for {image_name}")
    
    def _process_image(self, image_path: str):
        """Crop the image and return the cropped image, bounding box, and full-size image."""
        cropped_image_bbox, bbox, image_full_size = self._crop_image_bbox(image_path)
        return cropped_image_bbox, bbox, image_full_size

    def _predict_mask(self, cropped_image: np.ndarray):
        """Perform segmentation inference on the cropped image."""
        pil_image = Image.fromarray(cropped_image)
        image_tensor = self.transform(pil_image).float().to(device).unsqueeze(0)

        pred_mask = self.seg_model(image_tensor)
        pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        return (pred_mask > 0).float().numpy()

    def _resize_and_pad_mask(self, pred_mask: np.ndarray, bbox: tuple, full_size: tuple):
        """Resize the predicted mask and pad it to the original image size."""
        x_min, y_min, x_max, y_max = bbox
        cropped_height, cropped_width = y_max - y_min, x_max - x_min

        resized_mask = cv2.resize(pred_mask, (cropped_width, cropped_height))
        padded_mask = np.zeros(full_size, dtype=np.uint8)
        padded_mask[y_min:y_max, x_min:x_max] = resized_mask
        return padded_mask

    def _save_full_size_mask(self, mask: np.ndarray, image_name: str):
        """Save the full-size mask to disk."""
        mask_path = self.masks_dir / f"{image_name}.png"
        cv2.imwrite(str(mask_path), mask * 255)
        log.info(f"Full-size mask saved for {image_name} at {mask_path}")

    def _save_overlay_image(self, mask: np.ndarray, original_image: np.ndarray, image_name: str):
        """Save the overlay of the predicted mask on the original image."""
        full_size_height, full_size_width = original_image.shape[:2]
        colored_overlay = np.zeros_like(original_image)
        colored_overlay[:, :, 1] = 255  # Green overlay

        colored_mask = cv2.bitwise_and(colored_overlay, colored_overlay, mask=mask)
        alpha = 0.75
        overlay_image = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
        overlay_image = cv2.resize(overlay_image, (full_size_width // 10, full_size_height // 10))

        overlay_path = self.img_mask_comparison_dir / f"{image_name}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay_image)
        log.info(f"Overlay image saved for {image_name} at {overlay_path}")

    def process_directory(self):
        """
        Processes all images in the test directory for segmentation inference.
        """
        log.info(f"Processing images in directory: {self.test_dir}")
        for img_path in tqdm(self.test_dir.rglob("*.jpg"), desc="Processing images"):
            self.infer_single_image(img_path)
        log.info("Inference completed.")

def main(cfg: DictConfig):
    """
    Main function to initialize and start inference.

    Args:
        cfg (DictConfig): Configuration object.
    """
    trainer = UNetInference(cfg)
    trainer.process_directory()
