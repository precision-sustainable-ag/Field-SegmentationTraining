"""
UNet Segmentation Inference

This script defines a UNetInference class that loads a pre-trained UNet model and performs image segmentation
on a batch of images using bounding boxes provided in accompanying JSON metadata. It handles both standard and
tile-based inference (for large images), post-processes masks to match original image sizes, and saves the 
resulting cutouts and masks to disk.

"""
import cv2
import json
import torch
import logging
import numpy as np

from PIL import Image
from pathlib import Path
from datetime import datetime
import segmentation_models_pytorch as smp
from torchvision import transforms

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetInference:
    """
    Class to perform semantic segmentation inference using a pre-trained UNet model from segmentation_models_pytorch.
    """

    def __init__(self, image_dir: Path, trained_model_path: Path):
        """
        Initializes the UNetInference class and loads the trained UNet model.

        Args:
            image_dir (Path): Path to the root directory containing images.
            trained_model_path (Path): Path to the trained model (.pth) file.
        """
        logging.info(f"Initializing UNetInference at {datetime.now()}")
        
        self.image_dir = image_dir
        self.developed_images_dir = self.image_dir / "developed-images"
        self.cutout_dir = self.image_dir / "cutouts"
        self.trained_model_path = trained_model_path
        
        # Load pre-trained UNet model
        self.seg_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(DEVICE)

        self.seg_model.load_state_dict(torch.load(self.trained_model_path, map_location=DEVICE, weights_only=True))
        self.seg_model.eval()

        # Image preprocessing transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def read_metadata(self, json_path):
        """
        Reads bounding box metadata from a JSON file.

        Args:
            json_path (Path): Path to the JSON file.

        Returns:
            dict or None: Parsed JSON data or None if file is missing.
        """
        if not json_path.exists():
            logging.warning(f"JSON file not found: {json_path}")
            return None

        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def _predict_mask(self, cropped_image: np.ndarray):
        """
        Predicts the binary segmentation mask for a given cropped image.

        Args:
            cropped_image (np.ndarray): Image cropped using bounding box.

        Returns:
            np.ndarray: Predicted binary mask.
        """
        pil_image = Image.fromarray(cropped_image)
        image_tensor = self.transform(pil_image).float().to(DEVICE).unsqueeze(0)

        try:
            logging.info(f"Predicting mask for image of shape: {image_tensor.shape}")
            pred_mask = self.seg_model(image_tensor)
            pred_mask = torch.sigmoid(pred_mask)  # Convert logits to probabilities
            pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
            pred_mask = (pred_mask > 0.5).float().numpy().squeeze(-1)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

        return pred_mask

    def _predict_mask_in_tiles(self, image: np.ndarray, overlap_pixels=250, max_tile_size=4500):
        """
        Splits a large image into overlapping tiles, predicts mask for each tile, and stitches them back.

        Args:
            image (np.ndarray): Original large image.
            overlap_pixels (int): Overlap to prevent seam artifacts.
            max_tile_size (int): Max size to keep GPU memory under control.

        Returns:
            np.ndarray: Stitched full-size mask.
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
        Resizes predicted mask to bounding box size and pads it to match original image dimensions.

        Args:
            pred_mask (np.ndarray): Predicted binary mask.
            bbox (dict): Dictionary with min/max coordinates.
            full_size (tuple): Original image shape.

        Returns:
            np.ndarray: Padded mask.
        """
        x_min, x_max = bbox["x_min"], bbox["x_max"]
        y_min, y_max = bbox["y_min"], bbox["y_max"]
        cropped_width = x_max - x_min
        cropped_height = y_max - y_min
        
        resized_mask = cv2.resize(pred_mask, (cropped_width, cropped_height))
        padded_mask = np.zeros(full_size[:2], dtype=np.uint8)
        padded_mask[y_min:y_max, x_min:x_max] = resized_mask

        return padded_mask

    def get_bbox_minmax(self, bbox: dict):
        """
        Computes bounding box min and max pixel coordinates.

        Args:
            bbox (dict): Bounding box in YOLO format [x, y, w, h].

        Returns:
            dict: Bounding box with keys x_min, x_max, y_min, y_max.
        """
        y_min, y_max = bbox[1], bbox[1] + bbox[3]
        x_min, x_max = bbox[0], bbox[0] + bbox[2]
        return {
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max
        }

    def pred_mask(self, cropped_image: np.ndarray):
        """
        Decides between tile-based or full-image segmentation based on image size.

        Args:
            cropped_image (np.ndarray): Image cropped to bounding box.

        Returns:
            np.ndarray: Predicted mask.
        """
        if cropped_image.shape[0] > 4000 or cropped_image.shape[1] > 4000:
            return self._predict_mask_in_tiles(cropped_image)
        return self._predict_mask(cropped_image)

    def save_image(self, img_path: str, image_cropped: np.ndarray, padded_cropped_mask: np.ndarray, final_cutout_rgb: np.ndarray):
        """
        Saves the cropped image, mask, and cutout to disk.

        Args:
            img_path (str): Path to original image.
            image_cropped (np.ndarray): Cropped image.
            padded_cropped_mask (np.ndarray): Final binary mask.
            final_cutout_rgb (np.ndarray): Image with only segmented regions retained.
        """
        stem = Path(img_path).stem
        cropout_name = f"{stem}.jpg"
        final_mask_name = f"{stem}_mask.png"
        cutout_name = f"{stem}.png"

        cv2.imwrite(str(self.cutout_dir / cropout_name), image_cropped.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(self.cutout_dir / final_mask_name), (padded_cropped_mask * 255).astype(np.uint8))
        cv2.imwrite(str(self.cutout_dir / cutout_name), final_cutout_rgb.astype(np.uint8))

    def process_image(self, input_paths):
        """
        Handles full pipeline for one image:
        - Read JSON metadata
        - Extract crop
        - Predict mask
        - Resize/pad mask
        - Save outputs

        Args:
            input_paths (tuple): (image_path, json_path)
        """
        image_path, json_path = input_paths
        logging.info(f"Processing image: {image_path}")

        metadata = self.read_metadata(json_path)
        bbox = metadata["bbox"]
        if bbox is None:
            logging.warning(f"No bounding box found for {image_path}. Skipping.")
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
        """
        Processes all .jpg images in the developed-images folder with corresponding .json files in cutouts folder.
        """
        images = sorted(list(self.developed_images_dir.glob("*.jpg")))
        logging.info(f"Processing {len(images)} images in directory: {self.developed_images_dir}.")
        
        for img_path in images:
            json_path = self.cutout_dir / f"{img_path.stem}.json"
            self.process_image((img_path, json_path))

# Script Execution
logging.info(f"Starting UNet segmentation inference.")
image_dir = Path("/home/nsingh27/Field-SegmentationTraining/mask_generation/data/image_processing_dir")
trained_model_path =  Path("/home/nsingh27/Field-SegmentationTraining/mask_generation/data/field-tools/models/unet/unet_segmentation.pth")
unet_inference = UNetInference(image_dir, trained_model_path)
unet_inference.process_directory()
logging.info(f"UNet segmentation inference complete.")