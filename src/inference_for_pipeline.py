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
import matplotlib.pyplot as plt

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

        self.save_mask = cfg.inference_for_pipeline.save_mask
        if self.save_mask:
            self.masks_dir = self.timestamp_inference_results_dir / "masks"
            self.masks_dir.mkdir(parents=True, exist_ok=True)

        self.overlay_comparison = cfg.inference_for_pipeline.overlay_comparison
        if self.overlay_comparison:
            self.img_mask_comparison_dir = self.timestamp_inference_results_dir / "img_mask_overlayed"
            self.img_mask_comparison_dir.mkdir(parents=True, exist_ok=True)

        self.side_by_side = cfg.inference_for_pipeline.side_by_side_comparison
        if self.side_by_side:
            self.img_mask_side_by_side_dir = self.timestamp_inference_results_dir / "img_mask_side_by_side"
            self.img_mask_side_by_side_dir.mkdir(parents=True, exist_ok=True)

        self.trained_model_path = cfg.paths.unet_segmentation_model
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

        if cropped_image_shape[0] < 4000 and cropped_image_shape[1] < 4000:
            log.info(f"Image size is small enough for direct processing.")
            pred_mask = self._predict_mask(cropped_image)
        else:
            log.info(f"Image size is too big for direct processing. Resizing and processing.")
            # pred_mask = self._predict_mask(cv2.resize(cropped_image, (int(cropped_image_shape[1] * 0.5), int(cropped_image_shape[0] * 0.5)))) # Resize to half
            pred_mask = self._process_image_in_tiles(cropped_image) # Process in tiles

        padded_mask = self._resize_and_pad_mask(pred_mask, bbox, image_full_size.shape[:2])

        if self.save_mask:
            self._save_full_size_mask(padded_mask, image_name)
        
        if self.overlay_comparison:
            self._save_overlay_image(padded_mask, image_full_size, image_name)
        
        if self.side_by_side:
            self._save_side_by_side_comparison(image_full_size, padded_mask, image_name)

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
        # Apply sigmoid to convert logits to probabilities (for binary)
        pred_mask = torch.sigmoid(pred_mask)
        
        pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred_mask = (pred_mask > 0.5).float().numpy()
        return pred_mask
    
    def _process_image_in_tiles(self, image: np.ndarray, overlap_pixels=500):
        """Process the image in tiles to avoid memory issues.
        Args:
            image (np.ndarray): The input image.
            overlap_pixels (int): The number of overlapping pixels between tiles.
        Returns:
            np.ndarray: The full-size mask for the input image.
        """
        height, width = image.shape[:2]
        tile_h, tile_w = height // 2, width // 2
        step_h, step_w = tile_h - overlap_pixels, tile_w - overlap_pixels

        print(f"Image shape: {image.shape}")
        print(f"step_h: {step_h}, step_w: {step_w}")
        print(f"tile_h: {tile_h}, tile_w: {tile_w}")

        full_mask = np.zeros((height, width), dtype=np.float32)

        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                y_end, x_end = min(y + tile_h, height), min(x + tile_w, width) # Calculate end coordinates for tile

                tile = image[y:y_end, x:x_end] # Extract tile from image
                tile_pred = self._predict_mask(tile) # Predict mask for tile
                tile_pred_sequeezed = tile_pred.squeeze() # Remove the channel dimension               

                full_mask[y:y_end, x:x_end] = np.maximum(full_mask[y:y_end, x:x_end], tile_pred_sequeezed) # Combine overlapping tiles by taking the maximum value

        return full_mask
   
    def _resize_and_pad_mask(self, pred_mask: np.ndarray, bbox: tuple, full_size: tuple):
        """Resize the predicted mask and pad it to the original image size."""
        x_min, y_min, x_max, y_max = bbox
        cropped_height, cropped_width = y_max - y_min, x_max - x_min

        resized_mask = cv2.resize(pred_mask, (cropped_width, cropped_height))
        padded_mask = np.zeros(full_size, dtype=np.uint8)
        padded_mask[y_min:y_max, x_min:x_max] = resized_mask
        return padded_mask

    def _save_side_by_side_comparison(self, image: np.ndarray, mask: np.ndarray, image_name: str):
        """
        Save a side-by-side comparison of the original image and the predicted mask.

        Args:
            image (np.ndarray): Original RGB image.
            mask (np.ndarray): Predicted binary mask.
            image_name (str): Name of the image file (without extension).
        """
        # Create a figure for the side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original RGB image
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title(f"Original Image: {image_name}")

        # Predicted mask
        axes[1].imshow(mask, cmap="gray")
        axes[1].axis("off")
        axes[1].set_title("Predicted Mask")
        plt.tight_layout()
        # Save the plot
        comparison_path = self.img_mask_side_by_side_dir / f"{image_name}_side_by_side.png"
        plt.savefig(comparison_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        log.info(f"Side-by-side comparison saved for {image_name} at {comparison_path}")
        
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
        images = sorted(list(self.test_dir.rglob("*.jpg")))
        for img_path in tqdm(images, desc="Processing images"):
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
