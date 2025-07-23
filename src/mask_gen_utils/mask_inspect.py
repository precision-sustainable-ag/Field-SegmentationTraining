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
import sqlite3
import datetime

# Logging configuration
log = logging.getLogger(__name__)


DB_PATH = "inspection.db"

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS images (
        image_id TEXT PRIMARY KEY,
        image_path TEXT,
        mask_path TEXT,
        refined_mask_path TEXT,
        initial_tag TEXT,
        final_tag TEXT,
        tags TEXT,      -- optional for backward compatibility or if you want a field for "all tags"
        status TEXT,
        reviewer TEXT,
        timestamp TEXT
    )
    ''')
    # Add columns if running on an old DB (safe for repeated runs)
    try:
        c.execute("ALTER TABLE images ADD COLUMN initial_tag TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE images ADD COLUMN final_tag TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def add_or_update_image(image_id, image_path, mask_path, refined_mask_path, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT OR IGNORE INTO images (
            image_id, image_path, mask_path, refined_mask_path, 
            initial_tag, final_tag, tags, status, reviewer, timestamp)
        VALUES (?, ?, ?, ?, '', '', '', 'pending', '', '')
    ''', (image_id, image_path, mask_path, refined_mask_path))
    conn.commit()
    conn.close()

def get_images_for_review(tag_filter=None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if tag_filter:
        q = "SELECT * FROM images WHERE tags LIKE ?"
        c.execute(q, (f"%{tag_filter}%",))
    else:
        q = "SELECT * FROM images"
        c.execute(q)
    rows = c.fetchall()
    conn.close()
    return rows

def update_image_tags(
    image_id, 
    initial_tag=None, 
    final_tag=None, 
    tags=None, 
    status=None, 
    reviewer='', 
    db_path=DB_PATH
):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    fields = []
    values = []
    if initial_tag is not None:
        fields.append("initial_tag=?")
        values.append(initial_tag)
    if final_tag is not None:
        fields.append("final_tag=?")
        values.append(final_tag)
    if tags is not None:
        fields.append("tags=?")
        values.append(tags)
    if status is not None:
        fields.append("status=?")
        values.append(status)
    fields.append("reviewer=?")
    values.append(reviewer)
    fields.append("timestamp=?")
    values.append(timestamp)
    values.append(image_id)
    q = f"UPDATE images SET {', '.join(fields)} WHERE image_id=?"
    c.execute(q, values)
    conn.commit()
    conn.close()

def update_tags_for_sample(image_id, user_tags, reviewer='', db_path=DB_PATH):
    """
    Update tags for a sample:
      - If 'good' is among the tags, update only final_tag to 'good'.
      - Otherwise, update initial_tag to the canonical form of the tags.
    """
    tags = set([t.lower() for t in user_tags if t])
    status = 'reviewed'

    if "good" in tags:
        # Only set final_tag to 'good'
        update_image_tags(
            image_id=image_id,
            final_tag="good",
            status=status,
            reviewer=reviewer,
            db_path=db_path
        )
    else:
        # Set initial_tag to all canonical tags except 'good'
        tag_str = ",".join(sorted(tags))
        update_image_tags(
            image_id=image_id,
            initial_tag=tag_str,
            status=status,
            reviewer=reviewer,
            db_path=db_path
        )
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
        self.db_path = "inspection.db"

        init_db(self.db_path)
        self._populate_db_with_images()

        self.voxel_inspection_results_dir = Path(cfg.paths.voxel_inspection_results_dir)
        self.voxel_inspection_results_dir.mkdir(parents=True, exist_ok=True)
        self.voxel_inspection_results_db = Path(cfg.paths.voxel_inspection_results_db)

        # Configuration parameters for FiftyOne Voxel
        self.port = cfg.mask_gen.inspect.port
        self.dataset_name = cfg.mask_gen.inspect.dataset_name
        self.dataset = None
        self.session = None

        self.canonical_mapping = cfg.mask_gen.canonical_tag_mapping

        # pipeline mode
        self.mode = cfg.mode
        self.inspect_cfg = cfg.mask_gen.inspect

    def _populate_db_with_images(self):
        # Scan all masks/images and add to db if missing
        for mask_path in self.mask_gen_cutout_dir.glob("*_mask.png"):
            image_name = mask_path.name.replace("_mask.png", ".jpg")
            image_path = str(self.mask_gen_cutout_dir / image_name)
            mask_path_str = str(mask_path)
            refined_path = str(self.refined_masks_dir_source / mask_path.name) if self.refined_masks_dir_source else ""
            add_or_update_image(
                image_id=image_name,
                image_path=image_path,
                mask_path=mask_path_str,
                refined_mask_path=refined_path,
                db_path=self.db_path
            )

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
    
    def normalize_tags(self, user_tags):
        """
        Maps a list of user tags to canonical tags using keywords.

        Args:
            user_tags (list of str): Tags as entered by user or FiftyOne UI.
            canonical_mapping (dict): {canonical: keyword}

        Returns:
            List of canonical tags (de-duplicated).
        """
        normalized = set()
        user_tags = [str(t).lower() for t in user_tags if t]
        for canonical, keyword in self.canonical_mapping.items():
            for user_tag in user_tags:
                if keyword in user_tag:
                    normalized.add(canonical)
        return list(normalized)

    def _load_samples(self, use_final_tag=False) -> list:
        """
        Loads FiftyOne samples from the database.
        - If use_final_tag=True, assigns tags from 'final_tag' if present and non-empty, otherwise from 'initial_tag'.
        - If use_final_tag=False, always uses 'initial_tag' for displayed tags.
        """
        db_rows = get_images_for_review(db_path=self.db_path)
        samples = []
        for row in db_rows:
            (
                image_id, image_path, mask_path, refined_mask_path,
                initial_tag, final_tag, tags, status, reviewer, timestamp
            ) = row

            # Skip if files are missing
            if not Path(mask_path).exists():
                log.warning(f"Mask file not found for {image_path}. Skipping.")
                continue

            initial_mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            sample = fo.Sample(filepath=image_path)
            sample["initial_mask"] = fo.Segmentation(mask=initial_mask_array)

            # Attach refined mask if present
            if refined_mask_path and Path(refined_mask_path).exists():
                refined_mask_array = np.array(Image.open(refined_mask_path).convert("L"), dtype=np.uint8)
                sample["refined_mask"] = fo.Segmentation(mask=refined_mask_array)

            # Determine which tag to show in UI
            if use_final_tag and final_tag:
                display_tags = [t.strip() for t in final_tag.split(",") if t.strip()]
            else:
                display_tags = [t.strip() for t in initial_tag.split(",") if t.strip()]

            if display_tags:
                sample.tags = display_tags

            samples.append(sample)
        log.info(f"Loaded {len(samples)} samples from database.")
        return samples
    
    
    def _create_dataset(self, samples=None) -> fo.Dataset:
        """
        Creates a FiftyOne dataset from samples defined in the SQLite DB.
        If dataset with same name exists, it will be deleted (optionally you can skip this for persistence).

        Args:
            samples (list): Ignored—samples are loaded from DB.

        Returns:
            fo.Dataset: The newly created FiftyOne dataset.
        """
        log.info(f"Creating dataset: {self.dataset_name}")
        
        if self.dataset_name in fo.list_datasets():
            log.info(f"Dataset '{self.dataset_name}' already exists. Load it.")
            # fo.load_dataset(self.dataset_name)
            fo.delete_dataset(self.dataset_name)
            

        # Query the DB for all relevant rows
        db_rows = get_images_for_review(db_path=self.db_path)
        fo_samples = []
        for row in db_rows:
            image_id, image_path, mask_path, refined_mask_path, initial_tag, final_tag, tags, status, reviewer, timestamp = row
            # Skip if image/mask is missing
            if not Path(image_path).exists() or not Path(mask_path).exists():
                log.warning(f"Image or mask not found for {image_id}. Skipping.")
                continue

            # Load masks as arrays
            initial_mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            sample = fo.Sample(filepath=image_path)
            sample["initial_mask"] = fo.Segmentation(mask=initial_mask_array)
            # Attach refined mask if present
            if refined_mask_path and Path(refined_mask_path).exists():
                refined_mask_array = np.array(Image.open(refined_mask_path).convert("L"), dtype=np.uint8)
                sample["refined_mask"] = fo.Segmentation(mask=refined_mask_array)
            # Attach tags from DB
            if tags:
                sample.tags = [t.strip() for t in tags.split(",") if t.strip()]
            fo_samples.append(sample)

        dataset = fo.Dataset(self.dataset_name)
        dataset.add_samples(fo_samples)
        return dataset


    def _get_voxel_dataset_map(self) -> dict:
        """
        Creates a dictionary of image_name -> user tags from self.dataset.

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

    def _handle_db(self, tag_field="initial_tag", reviewer=''):
        """
        After tagging in FiftyOne, update initial_tag or final_tag in DB for each sample.

        Args:
            tag_field (str): Which DB field to update, 'initial_tag' or 'final_tag'
            reviewer (str): Name or identifier of the reviewer (optional)
        """
        assert tag_field in ("initial_tag", "final_tag"), "tag_field must be 'initial_tag' or 'final_tag'"

        for sample in self.dataset:
            image_name = Path(sample.filepath).name
            canonical_tags = self.normalize_tags(sample.tags)
            update_tags_for_sample(
                image_id=image_name,
                user_tags=canonical_tags,
                reviewer=reviewer,
                db_path=self.db_path
            )

    def run_voxel_inspection(self, reviewer=''):
        """
        Runs the full inspection workflow:
        """
        samples = self._load_samples()
        self.dataset: fo.Dataset = self._create_dataset(samples)
        self.session = fo.launch_app(self.dataset, port=self.port)

        try:
            print("\nTag your images, then close the FiftyOne app or press Ctrl+C here to save.")
            self.session.wait()
        except KeyboardInterrupt:
            print("\nSession manually interrupted by user.")
        finally:
            self.session.refresh()
            self.session.close()
            self._update_sqlite_db_with_tags(reviewer=reviewer)
            print("Session closed.")

        # Save the voxel inspection results
        self._handle_db()
        log.info(f"Tags saved to database: {self.voxel_inspection_results_db}")

    def _update_sqlite_db_with_tags(self, reviewer='', tag_field="initial_tag"):
        """
        After the session, updates SQLite DB with tags for all samples in the dataset.
        """
        for sample in self.dataset:
            image_name = Path(sample.filepath).name
            canonical_tags = self.normalize_tags(sample.tags)
            update_tags_for_sample(
                image_id=image_name,
                user_tags=canonical_tags,
                reviewer=reviewer,
                db_path=self.db_path
            )

def main(cfg: DictConfig) -> None:
    """
    Entry point for launching the voxel inspection process.

    Args:
        cfg (DictConfig): A configuration object containing all necessary paths and parameters.
    """

    inspector = FiftyOneMaskInspector(cfg)
    inspector.run_voxel_inspection()
