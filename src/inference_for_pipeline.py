import os
import cv2
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from src.utils.unet import UNet
from omegaconf import DictConfig
from torchvision import transforms

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
        current_date = datetime.now()
        log.info(f"Initializing UNetInference at {current_date}")
        self.model_save_dir = Path(cfg.paths.model_save_dir)

        # Identify the latest model checkpoint folder
        self.latest_folder = self._get_latest_folder()
        if self.latest_folder:
            trained_model_path = os.path.join(
                self.model_save_dir, self.latest_folder, "project/weights/unet_segmentation.pth"
            )
            log.info(f"Trained UNet model located at: {trained_model_path}")
        else:
            log.error("No valid folder with a trained model was found.")
            raise FileNotFoundError("No valid folder with a trained model was found.")
        
        # Load YOLO model for target weed detection
        log.info("Loading YOLO model for target weed detection.")
        self.target_weed_detection_model = YOLO(cfg.paths.yolo_weed_detection_model)

        # Load UNet segmentation model
        log.info("Loading UNet model for segmentation.")
        self.seg_model = UNet(in_channels=3, num_classes=1).to(device)
        self.seg_model.load_state_dict(torch.load(trained_model_path, map_location=device))
        self.seg_model.eval()
        log.info("UNet model loaded and set to evaluation mode.")

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize image to 512x512
            transforms.ToTensor(),         # Convert image to a PyTorch tensor
        ])

        # Set directories for testing and results
        self.test_dir = Path(cfg.paths.test_dir)
        self.inference_results_dir = Path(cfg.paths.inference_results_dir)
        self.inference_results_dir.mkdir(parents=True, exist_ok=True)

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
                folder_date = folder_name.split('_')[1]  # Extract the date part
                folder_date_obj = datetime.strptime(folder_date, "%Y-%m-%d")
                difference = abs((folder_date_obj - datetime.now()).days)

                if difference < min_difference:
                    min_difference = difference
                    latest_folder = folder_name
            except (IndexError, ValueError):
                log.warning(f"Skipping folder '{folder_name}' due to invalid date format.")

        log.info(f"Latest folder identified: {latest_folder}")
        return latest_folder

    def _convert_to_rgb(self, image_path):
        """
        Converts the given image to RGB format.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Image as a numpy array in RGB format.
        """
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _detect_weeds(self, image_path):
        """
        Detects target weed in the given image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict or None: Bounding box coordinates if detection is successful, otherwise None.
        """
        log.debug("Starting weed detection.")
        image = self._convert_to_rgb(image_path)
        results = self.target_weed_detection_model(image)
        
        if not results or not results[0].boxes.xyxy.tolist():
            log.warning(f"No detection found in image: {image_path}")
            return None

        # Extract bounding box coordinates
        bbox = results[0].boxes.xyxy.tolist()[0]
        x_min, y_min, x_max, y_max = map(int, bbox)
        return x_min, y_min, x_max, y_max

    def _crop_image_bbox(self, image_path):
        """
        Crop the image using the detected bounding box.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Cropped image.
        """
        image = self._convert_to_rgb(image_path)
        bbox = self._detect_weeds(image_path)
        if bbox is None:
            log.error(f"Could not detect weed in image: {image_path}")
            raise ValueError("Weed detection failed.")
        
        x_min, y_min, x_max, y_max = bbox
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image

    def infer_single_image(self, image_path: str):
        """
        Performs segmentation mask prediction for a single image and saves the result.

        Args:
            image_path (str): Path to the input image.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_dir = self.inference_results_dir / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem
        log.info(f"Processing image: {image_path}")

        # Pre-process the image
        cropped_image = self._crop_image_bbox(image_path)
        pil_cropped_image = Image.fromarray(cropped_image) # Convert to PIL image because self.transform expects itw

        img = self.transform(pil_cropped_image).float().to(device).unsqueeze(0)

        # Predict segmentation mask
        pred_mask = self.seg_model(img)

        # Post-process for visualization
        img = img.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred_mask = (pred_mask > 0).float()

        # Save input image and predicted mask
        fig = plt.figure()
        for i, data in enumerate([img, pred_mask], start=1):
            plt.subplot(1, 2, i)
            plt.imshow(data, cmap="gray" if i == 2 else None)
        plt.savefig(results_dir / f"{image_name}_output.png")
        plt.close()
        log.info(f"Results saved for {image_name} at {results_dir}")

    def process_directory(self):
        """
        Processes all images in the test directory for segmentation inference.
        """
        log.info(f"Processing images in directory: {self.test_dir}")
        for img_path in tqdm(self.test_dir.rglob("*.jpg"), desc="Processing images"):
            self.infer_single_image(img_path)
    
def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and start training.

    Args:
        cfg (DictConfig): Configuration object.
    """
    trainer = UNetInference(cfg)
    trainer.process_directory()
