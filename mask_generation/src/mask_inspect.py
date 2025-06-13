# Voxel inspection and image tagging script

"""
Script to create and visualize a FiftyOne dataset for image segmentation tasks.

This script:
- Loads image and corresponding mask pairs
- Creates FiftyOne samples with segmentation masks
- Opens the FiftyOne App for interactive sample selection
- Saves the names of selected samples to a CSV file
"""

import fiftyone as fo
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class FiftyOneMaskInspector:
    """
    Class to create and manage a FiftyOne dataset for image segmentation tasks.
    
    This class:
    - Loads image and mask files
    - Adds initial and/or post-processing masks as segmentation fields
    - Launches an interactive FiftyOne app for manual image tagging
    - Saves selected sample tags to a CSV file
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the FiftyOneMaskInspector with configuration parameters.
        
        Args:
            cfg (DictConfig): Configuration containing paths and UI settings for the inspection process.
        """
        self.initial_mask_inspection_source = Path(cfg.paths.initial_mask_inspection_source)
        self.post_processing_mask_inspection_source = Path(cfg.paths.post_processing_mask_inspection_source)
        self.voxel_inspection_results_dir = Path(cfg.paths.voxel_inspection_results_dir)
        self.port = cfg.inspect.port
        self.dataset_name = cfg.inspect.dataset_name

    def load_samples(self) -> list:
        """
        Loads image and corresponding segmentation masks into FiftyOne samples.

        Returns:
            list: List of `fiftyone.core.sample.Sample` objects containing image paths and mask annotations.
        """
        samples = []
        for image_path in self.initial_mask_inspection_source.glob("*.jpg"):
            stem = image_path.stem
            initial_mask_path = self.initial_mask_inspection_source / f"{stem}_mask.png"

            if not initial_mask_path.exists():
                log.warning(f"Warning: Initial mask not found for {image_path.name}. Skipping.")
                continue

            # Load initial prediction masks
            initial_mask_array = np.array(Image.open(initial_mask_path).convert("L"), dtype=np.uint8)
            sample = fo.Sample(filepath=str(image_path))
            sample["initial masks"] = fo.Segmentation(mask=initial_mask_array)

            # Optionally load post-processed masks
            if self.post_processing_mask_inspection_source:
                post_processing_mask_path = self.post_processing_mask_inspection_source / f"{stem}.png"
                if post_processing_mask_path.exists():
                    post_processing_mask_array = np.array(Image.open(post_processing_mask_path).convert("L"), dtype=np.uint8)
                    sample["prediction"] = fo.Segmentation(mask=post_processing_mask_array)
                else:
                    log.warning(f"Note: Post processing mask not found for {image_path.name}.")
            samples.append(sample)

        log.info(f"Loaded {len(samples)} samples.")
        return samples

    def create_dataset(self, samples) -> fo.Dataset:
        """
        Creates a new FiftyOne dataset and adds samples.

        Args:
            samples (list): List of FiftyOne samples to include in the dataset.

        Returns:
            fiftyone.core.dataset.Dataset: The created FiftyOne dataset object.
        """
        log.info(f"Creating dataset: {self.dataset_name}")
        
        if self.dataset_name in fo.list_datasets():
            log.info(f"Dataset '{self.dataset_name}' already exists. Deleting it.")
            fo.delete_dataset(self.dataset_name)

        dataset = fo.Dataset(self.dataset_name)
        dataset.add_samples(samples)
        return dataset

    def export_tags_to_csv(self, output_csv) -> None:
        """
        Saves image filenames and associated tags to a CSV file.

        Args:
            output_csv (Path): Path where the CSV file will be saved.
        """
        rows = []
        for sample in self.dataset:
            tags_str = ",".join(sample.tags) if sample.tags else ""
            rows.append({"image_name": Path(sample.filepath).name, "tags": tags_str})
        
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        log.info(f"Exported image names and tags to {output_csv}")

    def run_voxel_inspection(self) -> None:
        """
        Launches the FiftyOne UI for interactive sample selection and tagging.
        Saves the resulting tags to a CSV file upon session termination.
        """
        samples = self.load_samples()
        self.dataset = self.create_dataset(samples)
        self.session = fo.launch_app(self.dataset, port=self.port)

        try:
            print("\n\nFollow these instructions in the FiftyOne app:\n\n" \
                "1. On the left bar, click on the LABELS tabs and select desired labels\n" \
                "2. Click on the box of each image to select samples of interest\n" \
                "3. Click on 'Tag samples or Labels' icon in the bar on top of images\n" \
                "4. Give the desired name to the selected samples\n" \
                "5. Click on 'ADD...' and then 'APPLY'\n" \
                "6. Repeat steps 2–5 if you have more tags to add\n" \
                "7. In your terminal, press Ctrl+C to stop the session and save the tags\n\n")
            self.session.wait()
        except KeyboardInterrupt:
            print("\nSession manually interrupted by user.")
        finally:
            self.session.refresh()
            self.session.close()
            print("Session closed.")

        output_csv_dir = self.voxel_inspection_results_dir / f"{self.dataset_name}_{datetime.now().strftime('%Y%m%d')}"
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = output_csv_dir / f"{self.dataset_name}_tags_{datetime.now().strftime('%H%M')}.csv"
        self.export_tags_to_csv(output_csv_path)

def main(cfg: DictConfig) -> None:
    """
    Main entry point for running the voxel mask inspector.

    Args:
        cfg (DictConfig): Configuration object for specifying paths and UI settings.
    """
    inspector = FiftyOneMaskInspector(cfg)
    inspector.run_voxel_inspection()
