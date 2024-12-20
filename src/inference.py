"""
Script for performing inference using a pre-trained UNet model for image segmentation.

This script initializes a UNet model, identifies the latest model checkpoint based on folder names, 
and processes images in a specified directory for segmentation. The results are saved as images 
showing input and predicted masks side by side.
"""
import os
import torch
import logging
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from src.utils.unet import UNet
from datetime import datetime
from omegaconf import DictConfig
from torchvision import transforms

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
        current_date = datetime.now()
        log.info(f"Current date: {current_date}")
        self.model_save_dir = Path(cfg.paths.model_save_dir)

        # Identify the latest model checkpoint folder
        self.latest_folder = self._get_latest_folder()
        if self.latest_folder:
            trained_model_path = os.path.join(
                self.model_save_dir, self.latest_folder, "project/weights/unet_segmentation.pth"
            )
            log.info(f"Trained model path: {trained_model_path}")
        else:
            raise FileNotFoundError("No valid folder with a trained model was found.")

        # Load the UNet model
        self.model = UNet(in_channels=3, num_classes=1).to(device)
        self.model.load_state_dict(torch.load(trained_model_path, map_location=device))
        self.model.eval()
        log.info("Model loaded and set to evaluation mode.")

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize image to 512x512
            transforms.ToTensor(),         # Convert image to a PyTorch tensor
        ])

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

        return latest_folder

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
        img = self.transform(Image.open(image_path)).float().to(device).unsqueeze(0)
        pred_mask = self.model(img)

        # Post-process for visualization
        img = img.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred_mask = (pred_mask > 0).float()

        # Save input image and predicted mask
        fig = plt.figure()
        for i, data in enumerate([img, pred_mask], start=1):
            plt.subplot(1, 2, i)
            plt.imshow(data, cmap="gray")
        plt.savefig(results_dir / f"{image_name}_output.png")
        plt.close()
        log.info(f"Saved results for {image_path} to {results_dir}")

    def process_directory(self):
        """
        Processes all images in the test directory for segmentation inference.
        """
        input_dir = self.test_dir
        for img_path in tqdm(input_dir.rglob("*.jpg"), desc="Processing images"):
            self.infer_single_image(img_path)

def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and start training.

    Args:
        cfg (DictConfig): Configuration object.
    """
    trainer = UNetInference(cfg)
    trainer.process_directory()
