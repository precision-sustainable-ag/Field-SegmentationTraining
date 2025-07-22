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

# Logging configuration
log = logging.getLogger(__name__)

class FiftyOneMaskInspector:
    """
    Manages image-mask datasets and interactive inspection using FiftyOne.

    Functionality:
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
        # Initialize required paths from the configuration
        self.mask_gen_cutout_dir = Path(cfg.paths.mask_gen_cutout_dir)
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
        self.inspect_cfg = cfg.mask_gen.inspect

        # load db if exists or create a new one
        self.df = pd.read_csv(self.voxel_inspection_results_db, index_col=False) if self.voxel_inspection_results_db.exists() else pd.DataFrame()

    def _get_mask_paths_from_db(self) -> list:
        """
        Gets image names from the voxel inspection results database.

        Returns:
            list: A list of image names with their corresponding mask names.
        """
        if not self.voxel_inspection_results_db.exists():
            return []

        image_names = []
        for only_include_tag in self.inspect_cfg.only_include_tags:
            # Ensure "initial_voxel_tag" column is treated as string and handle NaN values
            matched = self.df[self.df["initial_voxel_tag"].fillna("").astype(str).str.contains(only_include_tag)]["image_name"]
            image_names.extend(matched)

        mask_paths = []
        for image_name in image_names:
            image_name = Path(image_name)
            mask_path = self.mask_gen_cutout_dir / str(image_name).replace(".jpg", "_mask.png")
            if mask_path.exists():
                mask_paths.append(mask_path)
            else:
                log.warning(f"Mask file not found for {image_name}. Skipping.")
        
        return mask_paths

    def _create_sample_with_masks(self, mask_path: Path):
        """
        Creates a FiftyOne sample with initial and refined masks (if available).

        Args:
            mask_path (Path): Path to the initial mask image.

        Returns:
            fo.Sample or None: The sample object or None if the mask is invalid.
        """
        image_name = mask_path.name.replace("_mask.png", ".jpg")
        if not mask_path.exists():
            log.warning(f"Warning: Initial mask not found for {image_name}. Skipping.")
            return None

        initial_mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        image_path = self.mask_gen_cutout_dir / image_name
        sample = fo.Sample(filepath=str(image_path))
        sample["initial_mask"] = fo.Segmentation(mask=initial_mask_array)

        if self.refined_masks_dir_source:
            refined_mask_path = self.refined_masks_dir_source / f"{Path(image_name).stem}_mask.png"
            if refined_mask_path.exists():
                refined_mask_array = np.array(Image.open(refined_mask_path).convert("L"), dtype=np.uint8)
                sample["refined_mask"] = fo.Segmentation(mask=refined_mask_array)
            else:
                log.warning(f"Note: Refined mask not found for {image_path.name}.")

        return sample

    def _load_samples(self) -> list:
        """
        Loads images and their corresponding masks into FiftyOne samples.

        Returns:
            list: A list of `fiftyone.core.sample.Sample` objects with attached segmentation masks:
        """
        if self.inspect_cfg.only_include_tags and self.voxel_inspection_results_db.exists():
            mask_paths = self._get_mask_paths_from_db()
            if not mask_paths:
                raise ValueError("No mask paths found in the database. 'only_include_tags' is activated. Please check the voxel inspection results database.")
        else:
            mask_paths = sorted(self.mask_gen_cutout_dir.glob("*_mask.png"))
        
            if not mask_paths:
                raise ValueError("No mask paths found. Please check the source directories or database.")

        samples = []
        for mask_path in mask_paths:
            sample = self._create_sample_with_masks(mask_path)
            if sample:
                samples.append(sample)

        log.info(f"Loaded {len(samples)} samples.")
        return samples
    
    def _create_tag_map_from_db(self) -> dict:
        """
        Creates a dictionary mapping image names to their tags from the voxel inspection results database.

        Returns:
            dict: A dictionary where keys are image names and values are lists of tags.
        """
        tag_map = {}
        for _, row in self.df.iterrows():
            image_name = str(row["image_name"])
            if pd.notna(row["final_voxel_tag"]):
                tag = row["final_voxel_tag"]
            elif pd.notna(row["initial_voxel_tag"]):
                tag = str(row["initial_voxel_tag"])
            else:
                tag = ""
            tags = [t.strip() for t in tag.split(",") if t.strip()]
            tag_map[image_name] = tags
        return tag_map
    
    def _create_dataset(self, samples) -> fo.Dataset:
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
        tag_map = {}
        if self.voxel_inspection_results_db.exists():
            tag_map = self._create_tag_map_from_db()
            log.info(f"Loaded tags for {len(tag_map)} images from {self.voxel_inspection_results_db}")
        else:
            log.info(f"No existing tags found in {self.voxel_inspection_results_db}. Starting fresh.")

        # Assign tags to samples
        for sample in samples:
            image_name = Path(sample.filepath).name
            if image_name in tag_map:
                sample.tags = tag_map[image_name]

        dataset = fo.Dataset(self.dataset_name)
        dataset.add_samples(samples)
        return dataset

    def _load_existing_data(self) -> dict:
        """
        Loads the existing CSV into self.df and returns a map of existing data.
        
        Returns:
            dict: A dictionary mapping image names to their initial and final voxel tags.
        """
        existing_data = {}
        for _, row in self.df.iterrows():
            existing_data[row["image_name"]] = {
                "initial_voxel_tag": row["initial_voxel_tag"],
                "final_voxel_tag": row["final_voxel_tag"]
            }
        return existing_data

    def _update_existing_rows(self, current_data: dict) -> None:
        """
        Updates self.df based on current_data if tags have changed.
        
        Args:
            current_data (dict): A dictionary mapping image names to their new tags.
        """
        for idx, row in self.df.iterrows():
            image_name = row["image_name"]
            if image_name not in current_data:
                continue

            new_tags = current_data[image_name]
            old_tags = str(row["initial_voxel_tag"]) if pd.notna(row["initial_voxel_tag"]) else ""

            if old_tags == "good":
                continue  # Do not change 'good'
            elif new_tags == "good":
                self.df.at[idx, "final_voxel_tag"] = new_tags
            elif new_tags != old_tags and new_tags != "good":
                self.df.at[idx, "initial_voxel_tag"] = new_tags

    def _append_new_rows(self, current_data: dict, existing_data: dict) -> None:
        """
        Appends rows to self.df for any new image names not already present.

        Args:
            current_data (dict): A dictionary mapping image names to their tags.
            existing_data (dict): A dictionary mapping existing image names to their tags.
        """
        new_rows = []
        for image_name, tags_str in current_data.items():
            if image_name not in existing_data:
                new_rows.append({
                    "image_name": image_name,
                    "initial_voxel_tag": tags_str,
                    "final_voxel_tag": ""
                })
        if new_rows:
            self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)

    def _create_new_dataframe(self, current_data: dict) -> None:
        """
        Creates a new DataFrame from scratch when no CSV exists.
        Args:
            current_data (dict): A dictionary mapping image names to their tags."""
        self.df = pd.DataFrame([
            {
                "image_name": image_name,
                "initial_voxel_tag": tags_str,
                "final_voxel_tag": ""
            }
            for image_name, tags_str in current_data.items()
        ])

    def _get_current_dataset_map(self) -> dict:
        """
        Creates a dictionary of image_name -> tags from self.dataset.

        Returns:
            dict: A dictionary where keys are image filenames and values are comma-separated tag strings.
        """
        dataset_map = {}
        for sample in self.dataset:
            image_name = Path(sample.filepath).name
            tags = sample.tags if sample.tags else []
            tags_str = ",".join(tags)
            dataset_map[image_name] = tags_str
        return dataset_map

    def _save_tags_to_db(self, output_csv: Path) -> None:
        """
        Writes updated tagging information from the FiftyOne session back to the voxel inspection CSV.

        - Updates existing entries if tags have changed (except for 'good' which is preserved).
        - Appends any new image entries not already in the CSV.

        Args:
            output_csv (Path): Destination path for the updated voxel inspection CSV.
        """
        output_csv = Path(output_csv)
        current_data = self._get_current_dataset_map()

        if output_csv.exists():
            existing_data = self._load_existing_data()
            self._update_existing_rows(current_data)
            self._append_new_rows(current_data, existing_data)
        else:
            self._create_new_dataframe(current_data)

        self.df.to_csv(output_csv, index=False)

    def run_voxel_inspection(self) -> None:
        """
        Runs the full inspection workflow:

        - Loads samples from the filesystem.
        - Creates a FiftyOne dataset.
        - Launches the FiftyOne App for user-driven tagging.
        - Waits for the session to close (Ctrl+C).
        - Exports tags to a CSV in the results directory.
        TODO: improve docstring for whole script
        """
        samples = self._load_samples()
        self.dataset: fo.Dataset = self._create_dataset(samples)
        self.session = fo.launch_app(self.dataset, port=self.port)

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
        self._save_tags_to_db(self.voxel_inspection_results_db)
        log.info(f"Tags saved to database: {self.voxel_inspection_results_db}")

def main(cfg: DictConfig) -> None:
    """
    Entry point for launching the voxel inspection process.

    Args:
        cfg (DictConfig): A configuration object containing all necessary paths and parameters.
    """
    inspector = FiftyOneMaskInspector(cfg)
    inspector.run_voxel_inspection()
