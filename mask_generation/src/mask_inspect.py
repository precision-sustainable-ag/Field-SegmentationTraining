"""
Voxel Inspection and Image Tagging Script
-----------------------------------------

This script facilitates the inspection and tagging of image segmentation masks using the FiftyOne toolkit. It is designed for workflows that involve manual validation or selection of image-mask pairs, such as in medical imaging, 3D modeling, or computer vision QA processes.

Functionality:
- Loads `.jpg` images and their corresponding mask files (`_mask.png` for initial masks and `.png` for refined masks).
- Wraps these files into FiftyOne samples with labeled segmentation fields.
- Launches the FiftyOne App for interactive selection and tagging.
- Saves the selected image names and associated tags into a CSV file.
- Moves tagged "good" images to a long-term storage (LTS) location for future use.
"""

import fiftyone as fo
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
import logging
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
    - Move images tagged as 'good' to a directory in lts or test directory based on mode.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the inspector with configuration parameters.

        Args:
            cfg (DictConfig): Configuration object with the required fields:
        """
        # Initialize and validate all required paths from the configuration
        self.initial_mask_inspection_source = Path(cfg.paths.initial_mask_inspection_source)
        self.refined_masks_dir_source = Path(cfg.paths.refined_masks_dir)

        # for mode "run_pipeline"
        self.lts_good_images_masks_destination_dir = Path(cfg.paths.lts_good_images_masks_destination_dir)
        self.lts_good_images_masks_destination_dir.mkdir(parents=True, exist_ok=True)

        # for mode "test"
        self.test_good_images_masks_destination_dir = Path(cfg.paths.test_good_images_masks_destination_dir)
        self.test_good_images_masks_destination_dir.mkdir(parents=True, exist_ok=True)

        self.voxel_inspection_results_dir = Path(cfg.paths.voxel_inspection_results_dir)
        self.voxel_inspection_results_dir.mkdir(parents=True, exist_ok=True)

        self.voxel_inspection_results_csv = Path(cfg.paths.voxel_inspection_results_csv)

        # Other configuration parameters
        self.port = cfg.inspect.port
        self.dataset_name = cfg.inspect.dataset_name

        # pipeline mode
        self.mode = cfg.mode

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

    def move_good_images_masks_to_lts(self, initial_mask_source_dir: Path, refined_masks_source_dir: Path, dest_dir: str):
        """
        Moves images tagged as "good" along with their corresponding masks to a destination directory.

        This function reads voxel inspection results from a CSV file and identifies images marked with
        the tag "good". It then searches for corresponding masks, prioritizing refined masks if available,
        and moves both the image and its mask to a structured directory for training purposes.

        Args:
            initial_mask_source_dir (Path): Directory containing the original images and masks.
            refined_masks_source_dir (Path): Directory containing refined masks (if available).
            dest_dir (str): Path to the destination directory where images and masks will be moved.
        """
        df_voxel_results = pd.read_csv(self.voxel_inspection_results_csv)
        if df_voxel_results.empty:
            logging.exception(f"No voxel inspection results found in {self.voxel_inspection_results_csv}.")
            return

        # Extract images tagged "good" from the DataFrame
        good_images = df_voxel_results[df_voxel_results['tags'].str.contains("good", na=False)]['image_name'].tolist() 

        # Move images and masks (refined or initial masks) from source directory to the destination directory
        for image_name in good_images:
            source_image_path = Path(initial_mask_source_dir) / image_name  # Source image path
            # Check if the mask exists in the refined masks source directory first
            if image_name in refined_masks_source_dir.glob("*.png"):
                log.info(f"Refined mask found for {image_name}. Moving from refined masks source directory.")
                source_mask_path = Path(refined_masks_source_dir) / f"{Path(image_name).stem}_mask.png"  
            # If not found, use the initial mask source directory
            else:
                log.info(f"Refined mask not found for {image_name}. Using initial mask source directory.")
                source_mask_path = Path(initial_mask_source_dir) / f"{Path(image_name).stem}_mask.png" 

            dest_images_dir = Path(dest_dir) / "train_images" # Destination images directory
            dest_masks_dir = Path(dest_dir) / "train_masks" # Destination masks directory
            dest_images_dir.mkdir(parents=True, exist_ok=True)
            dest_masks_dir.mkdir(parents=True, exist_ok=True)

            dest_image_path = dest_images_dir / image_name
            dest_mask_path = dest_masks_dir / f"{Path(image_name).stem}_mask.png"

            shutil.move(str(source_image_path), str(dest_image_path))
            shutil.move(str(source_mask_path), str(dest_mask_path))
            log.info(f"Moved {image_name} and its mask to {dest_dir}")

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
                "4. Enter the desired tag name: 'good', 'bad', 'red_missing', 'white_missing', 'mat_present' or 'other'\n"
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
        self.export_tags_to_csv(self.voxel_inspection_results_csv)
        log.info(f"Tags saved to csv: {self.voxel_inspection_results_csv}")

def main(cfg: DictConfig) -> None:
    """
    Entry point for launching the voxel inspection process.

    Args:
        cfg (DictConfig): A configuration object containing all necessary paths and parameters.
    """
    inspector = FiftyOneMaskInspector(cfg)
    inspector.run_voxel_inspection()
