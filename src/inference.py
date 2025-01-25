import os
import csv
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig
from torchvision import transforms
from src.utils.unet import UNet

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        log.info(f"Initializing inference at {datetime.now()}")

        # Load model checkpoint
        self.trained_model_path = Path(cfg.paths.unet_segmentation_model)
        self.trained_model_name = self.trained_model_path.stem
        self.model = UNet(in_channels=3, num_classes=1).to(device)
        self.model.load_state_dict(torch.load(self.trained_model_path, map_location=device))
        self.model.eval()
        log.info("Model loaded and set to evaluation mode.")

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        # Directories for test images, masks, and results
        self.test_dir = Path(cfg.paths.test_dir)
        self.test_masks_dir = Path(cfg.paths.test_masks_dir)
        self.inference_results_dir = Path(cfg.paths.inference_results_dir)
        self.inference_results_dir.mkdir(parents=True, exist_ok=True)

        self.inference_results_dir_timestamp = Path(
            f"{self.inference_results_dir}/{self.trained_model_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        )
        self.inference_results_dir_timestamp.mkdir(parents=True, exist_ok=True)

        self.image_metrics = {}  # Dictionary to store IoU and Dice scores for each image

    def _calculate_metrics(self, predicted_mask, true_mask):
        """
        Calculate Intersection over Union (IoU) and Dice score between predicted and true masks.

        Args:
            predicted_mask (np.ndarray): Predicted binary mask.
            true_mask (np.ndarray): True binary mask.

        Returns:
            tuple: IoU and Dice score.
        """
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
        true_mask = (true_mask > 0.5).astype(np.uint8)

        intersection = np.logical_and(predicted_mask, true_mask).sum()
        union = np.logical_or(predicted_mask, true_mask).sum()

        iou = intersection / union if union != 0 else 0
        dice = (2 * intersection) / (predicted_mask.sum() + true_mask.sum()) if (predicted_mask.sum() + true_mask.sum()) != 0 else 0

        return iou, dice

    def _save_metrics_to_csv(self, metrics_path: Path):
        """
        Save IoU and Dice scores for each image to a CSV file.

        Args:
            metrics_path (Path): Path to save the CSV file.
        """
        csv_file = metrics_path / "metrics.csv"
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "IoU", "Dice Score"])

            for image_name, (iou, dice) in self.image_metrics.items():
                writer.writerow([image_name, iou, dice])

        log.info(f"Metrics saved to {csv_file}")

    def infer_single_image(self, image_path: str):
        """
        Perform segmentation inference for a single image and save the results.

        Args:
            image_path (str): Path to the input image.
        """
        # Load true mask
        true_mask_path = self.test_masks_dir / f"{Path(image_path).stem}.png"
        true_mask = Image.open(true_mask_path).convert("L")
        true_mask = true_mask.resize((512, 512))
        true_mask = np.array(true_mask)
        true_mask = (true_mask > 0).astype(np.uint8)

        # Load image and predict mask
        image_name = Path(image_path).stem
        img = self.transform(Image.open(image_path)).float().to(device).unsqueeze(0)
        pred_mask = self.model(img)

        # Post-process predicted mask
        pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().detach().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # Calculate metrics
        iou, dice = self._calculate_metrics(pred_mask, true_mask)
        self.image_metrics[image_name] = (iou, dice)

        # Save visualization
        plt.figure()
        for i, data in enumerate([true_mask, pred_mask], start=1):
            plt.subplot(1, 2, i)
            plt.title(["True Mask", "Predicted Mask"][i-1])
            plt.imshow(data, cmap="gray")
        plt.savefig(self.inference_results_dir_timestamp / f"{image_name}_output.png")
        plt.close()
        log.info(f"Results saved for {image_path}")

    def process_directory(self):
        """
        Perform segmentation inference for all images in the test directory.
        """
        for img_path in tqdm(self.test_dir.rglob("*.jpg"), desc="Processing images"):
            self.infer_single_image(img_path)

        metrics_path = self.inference_results_dir_timestamp / "metrics"
        metrics_path.mkdir(parents=True, exist_ok=True)

        self._save_metrics_to_csv(metrics_path)

        log.info("Inference complete.")

def main(cfg: DictConfig):
    """
    Initialize and execute the inference pipeline.

    Args:
        cfg (DictConfig): Configuration object.
    """
    inference = UNetInference(cfg)
    inference.process_directory()
