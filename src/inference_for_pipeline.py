import os
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
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

        # Set up directories
        self.model_save_dir = Path(cfg.paths.model_save_dir)
        self.test_dir = Path(cfg.paths.test_dir)
        self.inference_results_dir = Path(cfg.paths.inference_results_dir)
        self.inference_results_dir.mkdir(parents=True, exist_ok=True)

        # Find the latest trained model
        self.latest_folder = self._get_latest_folder()
        if not self.latest_folder:
            log.error("No valid folder with a trained model was found.")
            raise FileNotFoundError("No valid folder with a trained model was found.")

        trained_model_path = os.path.join(
            self.model_save_dir, self.latest_folder, "project/weights/unet_segmentation.pth"
        )
        log.info(f"Trained UNet model located at: {trained_model_path}")

        # Load YOLO model
        log.info("Loading YOLO model for target weed detection.")
        self.target_weed_detection_model = YOLO(cfg.paths.yolo_weed_detection_model)

        # Load UNet model
        log.info("Loading UNet model for segmentation.")
        self.seg_model = UNet(in_channels=3, num_classes=1).to(device)
        self.seg_model.load_state_dict(torch.load(trained_model_path, map_location=device))
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
                folder_date_obj = datetime.strptime(folder_date, "%Y-%m-%d")
                difference = abs((folder_date_obj - datetime.now()).days)
                if difference < min_difference:
                    min_difference = difference
                    latest_folder = folder_name
            except (IndexError, ValueError):
                log.warning(f"Skipping folder '{folder_name}' due to invalid date format.")

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
        Detects target weed in the given image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple or None: Bounding box coordinates (x_min, y_min, x_max, y_max) if detection is successful; otherwise, None.
        """
        log.debug("Starting weed detection.")
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
        timestamp = datetime.now().strftime("%Y%m%d_%H")

        # Create directories for saving results
        results_dir = self.inference_results_dir / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)

        masks_dir = results_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        img_mask_comparison_dir = results_dir / "img_mask_overlayed"
        img_mask_comparison_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem
        log.info(f"Processing image: {image_path}")

        # Crop image according to bbox and predict mask
        cropped_image_bbox, bbox, image_full_size = self._crop_image_bbox(image_path)

        # Resize cropped image for faster inference
        cropped_image_resized = cv2.resize(
            cropped_image_bbox, 
            (cropped_image_bbox.shape[1] // 4, cropped_image_bbox.shape[0] // 4)
        )

        pil_cropped_resized_image = Image.fromarray(cropped_image_resized)
        image_tensor = self.transform(pil_cropped_resized_image).float().to(device).unsqueeze(0)

        # Perform segmentation inference
        pred_mask = self.seg_model(image_tensor)
        pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred_mask = (pred_mask > 0).float().numpy()

        # Resize predicted mask to the size of the cropped_image_bbox
        pred_mask_cropped_size = cv2.resize(
            pred_mask, 
            (cropped_image_bbox.shape[1], cropped_image_bbox.shape[0])
        )

        x_min, y_min, x_max, y_max = bbox
        full_size_height, full_size_width = image_full_size.shape[:2]

        # Create a padded mask with the same size as the original image
        padded_mask = np.zeros((full_size_height, full_size_width), dtype=np.uint8)
        padded_mask[y_min:y_max, x_min:x_max] = pred_mask_cropped_size

        # Save the full-size mask 
        cv2.imwrite(str(masks_dir / f"{image_name}.png"), padded_mask * 255)
        log.info(f"Full-size mask saved for {image_name} at {masks_dir}")

        # Save the predicted mask overlayed on the original image
        colored_overlay = np.zeros_like(image_full_size)
        colored_overlay[:, :, 1] = 255

        # Apply the mask to the colored overlay
        colored_mask = cv2.bitwise_and(colored_overlay, colored_overlay, mask=padded_mask)

        # Combine the original image and the colored overlay using the predicted mask
        alpha = 0.5
        colored_overlay = cv2.addWeighted(image_full_size, 1, colored_mask, alpha, 0)
        colored_overlay = cv2.resize(colored_overlay, (full_size_width // 10, full_size_height // 10))

        # Save the overlayed image
        cv2.imwrite(str(img_mask_comparison_dir / f"{image_name}_overlay.png"), colored_overlay)

        log.info(f"Original image and predicted mask comparison saved for {image_name} at {img_mask_comparison_dir}")

    def process_directory(self):
        """
        Processes all images in the test directory for segmentation inference.
        """
        log.info(f"Processing images in directory: {self.test_dir}")
        for img_path in tqdm(self.test_dir.rglob("*.jpg"), desc="Processing images"):
            self.infer_single_image(img_path)


def main(cfg: DictConfig):
    """
    Main function to initialize and start inference.

    Args:
        cfg (DictConfig): Configuration object.
    """
    trainer = UNetInference(cfg)
    trainer.process_directory()
