"""
UNet Segmentation Inference Script
==================================
This script performs semantic segmentation on crop images using a pre-trained UNet model.
It reads images and bounding box metadata, crops the region of interest, predicts a segmentation mask, and saves the results.
"""
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetInference:
    """
    Class to perform semantic segmentation inference using a pre-trained UNet model.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize UNetInference with configuration for paths and model.

        Args:
            cfg (DictConfig): Configuration object containing paths to images and trained model.
        """
        log.info(f"Initializing UNetInference at {datetime.now()}")
        
        self.image_dir = Path(cfg.paths.project_maskgen_dir)
        self.developed_images_dir = self.image_dir / "developed-images"
        self.cutout_dir = self.image_dir / "cutouts"
        self.detections_csv = self.cutout_dir / "temp_db.csv"
        self.trained_model_path = Path(cfg.paths.unet_segmentation_model)

        if self.detections_csv.exists():
            self.df = pd.read_csv(self.detections_csv)
        else:
            raise FileNotFoundError(f"Detections CSV not found at {self.detections_csv}")
        
        # Load pre-trained UNet model
        self.seg_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(DEVICE)

        self.seg_model.load_state_dict(torch.load(self.trained_model_path, map_location=DEVICE, weights_only=True))
        self.seg_model.eval()

        self.transform = transforms.Compose([transforms.ToTensor()])

    def update_mask_path(self, image_name, mask_path):
        """Update DataFrame with mask path for a given image name."""
        idx = self.df[self.df["image_name"] == image_name].index
        if len(idx) == 1:
            self.df.at[idx[0], "initial_mask_path"] = mask_path
        elif len(idx) > 1:
            # Shouldn't happen, but just in case
            self.df.loc[self.df["image_name"] == image_name, "initial_mask_path"] = mask_path
        else:
            log.warning(f"No detection row found for {image_name}; can't update mask_path.")

    def get_bbox_minmax(self, bbox):
        y_min, y_max = int(bbox[1]), int(bbox[1] + bbox[3])
        x_min, x_max = int(bbox[0]), int(bbox[0] + bbox[2])
        return {
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max
        }
    
    def _predict_mask(self, cropped_image: np.ndarray):
        """
        Predict the segmentation mask for a cropped image.

        Args:
            cropped_image (np.ndarray): The cropped RGB image.

        Returns:
            np.ndarray: Binary segmentation mask.
        """
        pil_image = Image.fromarray(cropped_image)
        image_tensor = self.transform(pil_image).float().to(DEVICE).unsqueeze(0)

        try:
            log.info(f"Predicting mask for image of shape: {image_tensor.shape}")
            pred_mask = self.seg_model(image_tensor)
            pred_mask = torch.sigmoid(pred_mask).squeeze(0).cpu().detach().permute(1, 2, 0)
            pred_mask = (pred_mask > 0.5).float().numpy().squeeze(-1)
        except Exception as e:
            log.error(f"Error during prediction: {e}")
            raise

        return pred_mask

    def _predict_mask_in_tiles(self, image: np.ndarray, overlap_pixels=250, max_tile_size=4500):
        """
        Perform segmentation on large images using overlapping tiles.

        Args:
            image (np.ndarray): Original large image.
            overlap_pixels (int): Overlap between tiles to reduce artifacts.
            max_tile_size (int): Maximum tile size to control GPU memory.

        Returns:
            np.ndarray: Combined binary segmentation mask.
        """
        height, width = image.shape[:2]
        step_h = min(np.ceil(height / 2).astype(int), max_tile_size - overlap_pixels)
        step_w = min(np.ceil(width / 2).astype(int), max_tile_size - overlap_pixels)
        tile_h, tile_w = step_h + overlap_pixels, step_w + overlap_pixels

        pred_mask = np.zeros((height, width), dtype=np.float32)
        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                y_end, x_end = min(y + tile_h, height), min(x + tile_w, width)
                tile = image[y:y_end, x:x_end]
                tile_pred = self._predict_mask(tile)
                pred_mask[y:y_end, x:x_end] = np.maximum(pred_mask[y:y_end, x:x_end], tile_pred.squeeze())
        return pred_mask

    def _resize_and_pad_mask(self, pred_mask: np.ndarray, bbox: dict, full_size: tuple):
        """
        Resize and embed the predicted mask into the full-size image.

        Args:
            pred_mask (np.ndarray): Predicted mask from cropped image.
            bbox (dict): Bounding box with min and max coordinates.
            full_size (tuple): Original image dimensions.

        Returns:
            np.ndarray: Full-size binary mask.
        """
        x_min, x_max = bbox["x_min"], bbox["x_max"]
        y_min, y_max = bbox["y_min"], bbox["y_max"]
        cropped_width = x_max - x_min
        cropped_height = y_max - y_min
        
        resized_mask = cv2.resize(pred_mask, (cropped_width, cropped_height))
        padded_mask = np.zeros(full_size[:2], dtype=np.uint8)
        padded_mask[y_min:y_max, x_min:x_max] = resized_mask

        return padded_mask

    def pred_mask(self, cropped_image: np.ndarray):
        """
        Decide between tile-based and standard inference.

        Args:
            cropped_image (np.ndarray): Cropped image for segmentation.

        Returns:
            np.ndarray: Predicted binary mask.
        """
        if cropped_image.shape[0] > 4000 or cropped_image.shape[1] > 4000:
            return self._predict_mask_in_tiles(cropped_image)
        return self._predict_mask(cropped_image)

    def save_image(self, img_path: str, image_cropped: np.ndarray, padded_cropped_mask: np.ndarray, final_cutout_rgb: np.ndarray):
        """
        Save cropped image, segmentation mask, and cutout image.

        Args:
            img_path (str): Original image path.
            image_cropped (np.ndarray): Cropped RGB image.
            padded_cropped_mask (np.ndarray): Final mask for cropped region.
            final_cutout_rgb (np.ndarray): RGB cutout of segmented object.
        """
        stem = Path(img_path).stem
        cropout_name = f"{stem}_0.jpg"
        final_mask_name = f"{stem}_0_mask.png"
        cutout_name = f"{stem}_0.png"
        mask_rel_path = str((self.cutout_dir / final_mask_name).relative_to(self.image_dir))

        cv2.imwrite(str(self.cutout_dir / cropout_name), image_cropped.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(self.cutout_dir / final_mask_name), (padded_cropped_mask * 255).astype(np.uint8))
        cv2.imwrite(str(self.cutout_dir / cutout_name), final_cutout_rgb.astype(np.uint8))

        # Update mask path in the DataFrame
        self.update_mask_path(Path(img_path).name, mask_rel_path)

    def process_row(self, row: pd.Series):
        image_name = row["image_name"]
        image_path = self.developed_images_dir / image_name
        log.info(f"Processing image: {image_path}")

        # Get bbox directly from DataFrame columns
        try:
            bbox = [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]]
            if any(pd.isna(bbox)):
                log.warning(f"No bounding box found for {image_name}. Skipping.")
                return
        except Exception as e:
            log.warning(f"Error reading bbox for {image_name}: {e}")
            return

        bx = self.get_bbox_minmax(bbox)
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        image_cropped = image[bx["y_min"]:bx["y_max"], bx["x_min"]:bx["x_max"]]
        pred_mask = self.pred_mask(image_cropped)
        padded_mask = self._resize_and_pad_mask(pred_mask, bx, image.shape[:2])
        padded_cropped_mask = padded_mask[bx["y_min"]:bx["y_max"], bx["x_min"]:bx["x_max"]]

        padded_mask_3d = np.repeat(padded_mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        cropped_padded_mask = padded_mask_3d[bx["y_min"]:bx["y_max"], bx["x_min"]:bx["x_max"]]
        image_cropped_bgr = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
        final_cutout_rgb = np.where(cropped_padded_mask == 1, image_cropped_bgr, 0)

        self.save_image(image_path, image_cropped_bgr, padded_cropped_mask, final_cutout_rgb)

    def process_directory(self):
        for _, row in self.df.iterrows():
            self.process_row(row)
        
        # Save updated DataFrame (with mask_path column) back to CSV
        self.df.to_csv(self.detections_csv, index=False)

def main(cfg: DictConfig) -> None:
    """
    Entry point for running the UNet segmentation inference pipeline.

    Args:
        cfg (DictConfig): Configuration with paths to input image directory and trained model.
    """
    log.info(f"Starting UNet segmentation inference.")
    unet_inference = UNetInference(cfg)
    unet_inference.process_directory()
    log.info(f"UNet segmentation inference complete.")