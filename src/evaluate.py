import os
import yaml
import torch
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig
from torchvision import transforms
from src.utils.unet import UNet
from src.utils import custom_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class UNetInference:
    """
    A class to perform segmentation inference using a pre-trained UNet model.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the UNetInference class, load the model, and set up directories.

        Args:
            cfg (DictConfig): Configuration object containing paths and settings.
        """
        log.info(f"Initializing inference at {datetime.now()}")

        # Load the trained model
        self.trained_model_path = Path(cfg.paths.unet_segmentation_model)
        self.trained_model_name = self.trained_model_path.parent.parent.name

        self.model = UNet(in_channels=3, num_classes=1).to(device)
        self.model.load_state_dict(torch.load(self.trained_model_path, map_location=device, weights_only=True))
        self.model.eval()
        log.info("Model loaded and set to evaluation mode.")

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        # Directories for test data and results
        self.data_dir = Path(cfg.paths.data_dir)
        self.test_dir = Path(cfg.paths.test_dir)
        self.test_masks_dir = Path(cfg.paths.test_masks_dir)
        self.inference_results_dir = Path(cfg.paths.inference_results_dir)
        self.inference_results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        self.results_dir_with_timestamp = self.inference_results_dir / f"{self.trained_model_name}_{timestamp}/predictions"
        self.results_dir_with_timestamp.mkdir(parents=True, exist_ok=True)

        # Persistent data
        self.image_metrics = {}
        self.persistent_table = pd.read_csv(cfg.paths.persistent_table_path, low_memory=False)

    def _calculate_metrics(self, predicted_mask: np.ndarray, true_mask: np.ndarray) -> tuple:
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
    
    def _save_metrics(self):
        """
        Save IoU and Dice metrics per species and overall.
        """
        log.info("Calculating and saving metrics...")
        combined_species_df, species_metrics = self._data_by_speceis()

        # Calculate overall metrics
        overall_metrics = combined_species_df[['iou', 'dice_score']].mean()
        overall_row = pd.DataFrame({
            'mean_iou': [overall_metrics['iou']],
            'mean_dice': [overall_metrics['dice_score']]
        }, index=['Overall'])

        # Calculate total images
        total_images = species_metrics['count'].sum()
        total_images_row = pd.DataFrame({
            'count': [total_images]
        }, index=['Total Images'])


        species_metrics = pd.concat([species_metrics, overall_row, total_images_row])

        species_metrics.index.name = 'species' # Set index name to species

        # Save metrics to CSV
        csv_save_dir = self.results_dir_with_timestamp.parent / "metrics_dir"
        csv_save_dir.mkdir(parents=True, exist_ok=True)

        output_path = csv_save_dir / "species_metrics.csv"
        species_metrics.to_csv(output_path)
        log.info(f"Metrics saved to {output_path}")

    def _data_by_speceis(self):
        """
        Filter persistent table for relevant image data and calculate per-species metrics.

        Returns:
            tuple: Combined species DataFrame and per-species metrics DataFrame.
        """
        # Create DataFrame from image metrics
        metrics_df = pd.DataFrame.from_dict(self.image_metrics, orient='index', columns=['iou', 'dice_score'])
        metrics_df.index.name = 'Stem'

        # Filter persistent table for relevant image data
        df = self.persistent_table
        filtered_df = df[df['Extension'] == 'jpg'][['Stem', 'Species']].dropna()
        combined_species_df = pd.merge(filtered_df, metrics_df, on='Stem')

        # Calculate per-species and overall metrics
        species_metrics = combined_species_df.groupby('Species').agg(
            count=('iou', 'count'),
            mean_iou=('iou', 'mean'), 
            mean_dice=('dice_score', 'mean')
        )

        return combined_species_df, species_metrics

    def _save_visualization(self, img_np: np.ndarray, pred_mask: np.ndarray, image_name: str, species_dir: Path):
        """
        Save a side-by-side visualization of the input image and predicted mask.

        Args:
            img_np (np.ndarray): Input image as a numpy array.
            pred_mask (np.ndarray): Predicted mask.
            image_name (str): Name of the input image.
            species_dir (Path): Directory to save the visualization.
        """
        plt.figure(figsize=(20, 10))
        plt.suptitle(f"{image_name}: {species_dir.name}")
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(img_np)

        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask, cmap="gray")

        output_path = species_dir/f"{image_name}_prediction.png"
        plt.savefig(output_path)
        plt.close()
        log.info(f"Visualization saved for {image_name}")

    def _copy_unet_conf_to_inference_dir(self):
        """
        Copy the UNet configuration file to the inference results directory.
        """
        unet_conf_path = self.trained_model_path.parent.parent / "unet_conf.yaml"
        inference_conf_path = self.results_dir_with_timestamp.parent / "unet_conf.yaml"
        os.system(f"cp {unet_conf_path} {inference_conf_path}")

    def infer_single_image(self, image_path: str):
        """
        Perform segmentation inference for a single image and save the results alongwith the unet config of model.

        Args:
            image_path (str): Path to the input image.
        """
        image_name = Path(image_path).stem

        # Load true mask
        true_mask_path = self.test_masks_dir / f"{image_name}.png"
        true_mask = Image.open(true_mask_path).convert("L")
        true_mask = np.array(true_mask.resize((512, 512)))
        true_mask = (true_mask > 0).astype(np.uint8)

        # Load and preprocess image
        img = self.transform(Image.open(image_path)).float().to(device).unsqueeze(0)

        # Convert img to numpy array
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        # Predict mask
        pred_mask = self.model(img).squeeze(0).squeeze(0).cpu().detach().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # Calculate metrics
        iou, dice = self._calculate_metrics(pred_mask, true_mask)
        self.image_metrics[image_name] = (iou, dice)

        # Save visualization by species
        combined_species_df, _ = self._data_by_speceis()

        for stem in combined_species_df['Stem']:
            if image_name in stem:
                species = combined_species_df[combined_species_df['Stem'] == stem]['Species'].values[0]
                species_dir = self.results_dir_with_timestamp / species
                species_dir.mkdir(parents=True, exist_ok=True)
                self._save_visualization(img_np, pred_mask, image_name, species_dir)

    def process_directory(self):
        """
        Perform segmentation inference for all images in the test directory.
        """
        image_paths = list(self.test_dir.rglob("*.jpg"))

        with tqdm(total=len(image_paths), desc="Processing images", unit="image") as pbar:
            for img_path in image_paths:
                self.infer_single_image(img_path)
                pbar.update(1)
        
        self._save_metrics()
        log.info("Metrics saved.")
        self._copy_unet_conf_to_inference_dir()
        log.info("UNet configuration copied to inference directory.")
        log.info("Inference completed.")

def main(cfg: DictConfig):
    """
    Initialize and execute the inference pipeline.

    Args:
        cfg (DictConfig): Configuration object.
    """
    inference = UNetInference(cfg)
    inference.process_directory()
