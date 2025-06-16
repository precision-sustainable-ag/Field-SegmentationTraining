"""
Voxel Inspection and Image Tagging Script
-----------------------------------------

This script facilitates the inspection and tagging of image segmentation masks using the FiftyOne toolkit. It is designed for workflows that involve manual validation or selection of image-mask pairs, such as in medical imaging, 3D modeling, or computer vision QA processes.

Functionality:
- Loads `.jpg` images and their corresponding mask files (`_mask.png` for initial masks and `.png` for refined masks).
- Wraps these files into FiftyOne samples with labeled segmentation fields.
- Launches the FiftyOne App for interactive selection and tagging.
- Saves the selected image names and associated tags into a CSV file.
- Optionally moves tagged "good" images to a long-term storage (LTS) location for future use.
"""

import fiftyone as fo
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
import os
import shutil

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

class FiftyOneMaskInspector:
    """
    Manages image-mask datasets and interactive inspection using FiftyOne.

    Responsibilities:
    - Load image-mask pairs from specified directories.
    - Create and manage FiftyOne datasets.
    - Launch FiftyOne UI for visual inspection and tagging.
    - Export tagging results to a CSV.
    - Optionally move selected images (e.g., tagged as 'good') to a directory in lts.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the inspector with configuration parameters.

        Args:
            cfg (DictConfig): Configuration object with the required fields:
        """
        self.initial_mask_inspection_source = Path(cfg.paths.initial_mask_inspection_source)
        self.refined_masks_dir_source = Path(cfg.paths.refined_masks_dir)
        self.voxel_inspection_results_dir = Path(cfg.paths.voxel_inspection_results_dir)
        self.port = cfg.inspect.port
        self.dataset_name = cfg.inspect.dataset_name

    def load_samples(self) -> list:
        """
        Loads images and their corresponding masks into FiftyOne samples.

        Returns:
            list: A list of `fiftyone.core.sample.Sample` objects with attached segmentation masks:
                  - "initial masks": from `_mask.png` files
                  - "prediction": from refined_masks `.png` files, if available
        """
        samples = []
        for image_path in self.initial_mask_inspection_source.glob("*.jpg"):
            stem = image_path.stem
            initial_mask_path = self.initial_mask_inspection_source / f"{stem}_mask.png"

            if not initial_mask_path.exists():
                log.warning(f"Warning: Initial mask not found for {image_path.name}. Skipping.")
                continue

            initial_mask_array = np.array(Image.open(initial_mask_path).convert("L"), dtype=np.uint8)
            sample = fo.Sample(filepath=str(image_path))
            sample["initial masks"] = fo.Segmentation(mask=initial_mask_array)

            if self.refined_masks_dir_source:
                refined_mask_path = self.refined_masks_dir_source / f"{stem}_mask.png"
                if refined_mask_path.exists():
                    refined_mask_array = np.array(Image.open(refined_mask_path).convert("L"), dtype=np.uint8)
                    sample["prediction"] = fo.Segmentation(mask=refined_mask_array)
                else:
                    log.warning(f"Note: Refined mask not found for {image_path.name}.")
            samples.append(sample)

        log.info(f"Loaded {len(samples)} samples.")
        return samples

    def create_dataset(self, samples) -> fo.Dataset:
        """
        Creates a FiftyOne dataset with the given samples.

        If a dataset with the same name exists, it will be deleted before creation.

        Args:
            samples (list): A list of FiftyOne samples.

        Returns:
            fo.Dataset: The newly created FiftyOne dataset.
        """
        log.info(f"Creating dataset: {self.dataset_name}")
        if self.dataset_name in fo.list_datasets():
            log.info(f"Dataset '{self.dataset_name}' already exists. Deleting it.")
            fo.delete_dataset(self.dataset_name)

        dataset = fo.Dataset(self.dataset_name)
        dataset.add_samples(samples)
        return dataset

    def export_tags_to_csv(self, output_csv: Path) -> None:
        """
        Exports the image filenames and their associated tags to a CSV file.

        Args:
            output_csv (Path): Destination path for the CSV file.
        """
        rows = []
        for sample in self.dataset:
            tags_str = ",".join(sample.tags) if sample.tags else ""
            rows.append({"image_name": Path(sample.filepath).name, "tags": tags_str})

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)

    def move_good_images_masks_to_lts(self, source_dir: str, image_names_csv: str, dest_dir: str):
        """
        Moves images tagged as 'good' from a source directory to a long-term storage directory.

        Args:
            source_dir (str): Path to the directory containing the original images.
            image_names_csv (str): Path to a CSV file with 'image_name' and 'tag' columns.
            dest_dir (str): Destination directory for moving selected images.
        """
        df = pd.read_csv(image_names_csv, header=None, names=["image_name", "tag"])
        good_images = df[df["tag"] == "good"]["image_name"].tolist()

        os.makedirs(dest_dir, exist_ok=True)

        for image_name in good_images:
            source_image_path = os.path.join(source_dir, image_name)
            dest_image_path = os.path.join(dest_dir, image_name)

            if os.path.exists(source_image_path):
                shutil.move(source_image_path, dest_image_path)
                print(f"Moved {image_name} to {dest_dir}")
            else:
                print(f"Image {image_name} not found in {source_dir}")

    def run_voxel_inspection(self) -> None:
        """
        Runs the full inspection workflow:

        - Loads samples from the filesystem.
        - Creates a FiftyOne dataset.
        - Launches the FiftyOne App for user-driven tagging.
        - Waits for the session to close (Ctrl+C).
        - Exports tags to a CSV in the results directory.
        """
        samples = self.load_samples()
        self.dataset = self.create_dataset(samples)
        self.session = fo.launch_app(self.dataset, port=self.port)

        try:
            print("\n\nFollow these instructions in the FiftyOne app:\n\n"
                "1. On the left bar, click on the LABELS tab and select desired labels\n"
                "2. Click on the box of each image to select samples (images) of interest\n"
                "3. Click on 'Tag samples or Labels' icon in the bar above the samples\n"
                "4. Enter the desired tag name: 'good', 'red_missing', 'white_missing', or 'other'\n"
                "5. Click 'ADD...' and then 'APPLY'\n"
                "6. Repeat steps 2–5 for additional tags\n"
                "7. Press Ctrl+C in the terminal to end the session and save the tags\n")
            self.session.wait()
        except KeyboardInterrupt:
            print("\nSession manually interrupted by user.")
        finally:
            self.session.refresh()
            self.session.close()
            print("Session closed.")

        # Save the voxel inspection results
        self.voxel_inspection_results_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = self.voxel_inspection_results_dir / f"{self.dataset_name}.csv"
        self.export_tags_to_csv(output_csv_path)
        log.info(f"Tags saved to csv: {output_csv_path}")

def main(cfg: DictConfig) -> None:
    """
    Entry point for launching the voxel inspection process.

    Args:
        cfg (DictConfig): A configuration object containing all necessary paths and parameters.
    """
    inspector = FiftyOneMaskInspector(cfg)
    inspector.run_voxel_inspection()
