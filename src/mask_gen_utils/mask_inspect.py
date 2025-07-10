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

        self.voxel_inspection_results_dir = Path(cfg.paths.voxel_inspection_results_dir)
        self.voxel_inspection_results_dir.mkdir(parents=True, exist_ok=True)
        self.voxel_inspection_results_db = Path(cfg.paths.voxel_inspection_results_db)

        # Configuration parameters for FiftyOne Voxel
        self.port = cfg.mask_gen.inspect.port
        self.dataset_name = cfg.mask_gen.inspect.dataset_name
        self.dataset = None
        self.session = None

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
            sample["initial_mask"] = fo.Segmentation(mask=initial_mask_array)

            if self.refined_masks_dir_source:
                refined_mask_path = self.refined_masks_dir_source / f"{stem}_mask.png"
                if refined_mask_path.exists():
                    refined_mask_array = np.array(Image.open(refined_mask_path).convert("L"), dtype=np.uint8)
                    sample["refined_mask"] = fo.Segmentation(mask=refined_mask_array)
                else:
                    log.warning(f"Note: Refined mask not found for {image_path.name}.")
            samples.append(sample)

        log.info(f"Loaded {len(samples)} samples.")
        return samples

    def create_dataset(self, samples) -> fo.Dataset:
        """
        Creates a FiftyOne dataset with the given samples.

        If a dataset with the same name exists, it will be deleted before creation.
        Loads tags from self.voxel_inspection_results_db if available.

        Args:
            samples (list): A list of FiftyOne samples.

        Returns:
            fo.Dataset: The newly created FiftyOne dataset.
        """
        log.info(f"Creating dataset: {self.dataset_name}")
        if self.dataset_name in fo.list_datasets():
            log.info(f"Dataset '{self.dataset_name}' already exists. Deleting it.")
            fo.delete_dataset(self.dataset_name)

        # Load tags from CSV if available
        tags_map = {}
        if self.voxel_inspection_results_db.exists():
            df = pd.read_csv(self.voxel_inspection_results_db)
            for _, row in df.iterrows():
                image_name = str(row["image_name"])
                tags_str = str(row["voxel tags"]) if pd.notna(row["voxel tags"]) else ""
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                tags_map[image_name] = tags
            log.info(f"Loaded tags for {len(tags_map)} images from {self.voxel_inspection_results_db}")
        else:
            log.info(f"No existing tags found in {self.voxel_inspection_results_db}. Starting fresh.")

        # Assign tags to samples
        for sample in samples:
            image_name = Path(sample.filepath).name
            if image_name in tags_map:
                sample.tags = tags_map[image_name]

        dataset = fo.Dataset(self.dataset_name)
        dataset.add_samples(samples)
        return dataset

    def save_tags_to_db(self, output_csv: Path) -> None:
        """
        Updates a CSV file with image filenames and their associated tags from self.dataset.
        Only replaces tags if they have changed; otherwise, keeps the existing tags.
        Appends new images if not already present.
        """
        output_csv = Path(output_csv)

        # Create a dictionary from the current dataset
        current_data = {
            Path(sample.filepath).name: ",".join(sample.tags) if sample.tags else ""
            for sample in self.dataset
        }

        if output_csv.exists():
            # Load existing CSV
            df = pd.read_csv(output_csv)

            # Create a map from the current CSV
            existing_data = dict(zip(df["image_name"], df["voxel tags"]))

            # Update only if tags have changed
            for idx, row in df.iterrows():
                image_name = row["image_name"]
                if image_name in current_data:
                    new_tags = current_data[image_name]
                    old_tags = str(row["voxel tags"]) if pd.notna(row["voxel tags"]) else ""
                    if new_tags and new_tags != old_tags:
                        df.at[idx, "voxel tags"] = new_tags
                    # If new_tags is empty or unchanged, keep the old tags

            # Append new rows (not in the original CSV)
            for image_name, tags_str in current_data.items():
                if image_name not in existing_data:
                    df = pd.concat([df, pd.DataFrame([{
                        "image_name": image_name,
                        "voxel tags": tags_str
                    }])], ignore_index=True)
        else:
            # Create new DataFrame if CSV doesn't exist
            df = pd.DataFrame([
                {"image_name": image_name, "voxel tags": tags_str}
                for image_name, tags_str in current_data.items()
            ])

        # Save the updated DataFrame
        df.to_csv(output_csv, index=False)

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
        self.session = fo.launch_app(self.dataset)

        try:
            print("\n\nFollow these instructions in the FiftyOne app:\n\n"
                "1. On the left bar, click on the LABELS tab and select desired labels: 'initial masks' or 'refined masks'\n"
                "2. On the left bar, click on the TAGS and then select 'sample tags'\n"
                "3. Click on the box of each image to select samples (images) of interest\n"
                "4. Click on 'Tag samples or Labels' icon in the bar above the samples\n"
                "5. Enter one of these tag names: 'good', 'bad', 'red_missing', 'white_missing', 'mat_present' or 'other'\n"
                "6. Click 'ADD...' and then 'APPLY'\n"
                "7. Repeat steps 2–6 for additional tags\n"
                "8. Press Ctrl+C in the terminal to end the session and save the tags\n")
            self.session.wait()
        except KeyboardInterrupt:
            print("\nSession manually interrupted by user.")
        finally:
            self.session.refresh()
            self.session.close()
            print("Session closed.")

        # Save the voxel inspection results
        self.save_tags_to_db(self.voxel_inspection_results_db)
        log.info(f"Tags saved to database: {self.voxel_inspection_results_db}")

def main(cfg: DictConfig) -> None:
    """
    Entry point for launching the voxel inspection process.

    Args:
        cfg (DictConfig): A configuration object containing all necessary paths and parameters.
    """
    inspector = FiftyOneMaskInspector(cfg)
    inspector.run_voxel_inspection()
