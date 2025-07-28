"""
Voxel Mask Inspection and Tagging Pipeline
------------------------------------------

This script provides an automated pipeline for the inspection, validation, and interactive tagging of image segmentation masks using the FiftyOne toolkit.

Key Features:
- Automatically detects and adds new image/mask pairs to a persistent SQLite database.
- Loads `.jpg` images and their corresponding mask files (`*_mask.png` for initial masks and optionally refined masks).
- Wraps each image and its mask(s) into FiftyOne samples for visualization and manual review.
- Launches the FiftyOne App for inspection and tagging of segmentation masks.
- Tracks image status and user-generated tags in the database.
- (Not implemented) Moves images tagged as 'good' to a long-term storage location as needed. TODO: Implement this in mask_refine.py.
"""

import os
import fiftyone as fo
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import sqlite3
import datetime
from typing import List, Dict, Set, Tuple

# Logging configuration
log = logging.getLogger(__name__)


TABLE_NAME = "mask_gen_images"

class InspectionDB:
    def __init__(self, db_path: str) -> None:
        log.info(f"Connecting to SQLite DB at {db_path}")
        try:
            self.conn = sqlite3.connect(db_path)
            self._init_db()
        except Exception as e:
            log.error(f"Failed to connect or initialize DB at {db_path}: {e}")
            raise

    def _init_db(self) -> None:
        c = self.conn.cursor()
        try:
            c.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                image_id TEXT PRIMARY KEY,
                image_path TEXT,
                mask_path TEXT,
                refined_mask_path TEXT,
                initial_tag TEXT,
                final_tag TEXT,
                tags TEXT,
                status TEXT,
                reviewer TEXT,
                timestamp TEXT,
                refine_params TEXT
            )
            ''')
            
            # Add refine_params if missing
            c.execute(f"PRAGMA table_info({TABLE_NAME})")
            existing_cols = [row[1] for row in c.fetchall()]
            if "refine_params" not in existing_cols:
                c.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN refine_params TEXT")

            self.conn.commit()
        except Exception as e:
            log.error(f"Error creating/updating DB table: {e}")
            raise
    
    def add_images_bulk(self, image_info_list: List[Tuple]) -> None:
        log.info(f"Bulk-inserting {len(image_info_list)} new images into DB")
        c = self.conn.cursor()
        try:
            c.executemany(f'''
                INSERT OR IGNORE INTO {TABLE_NAME} (
                    image_id, image_path, mask_path, refined_mask_path, 
                    initial_tag, final_tag, tags, status, reviewer, timestamp
                ) VALUES (?, ?, ?, ?, '', '', '', 'pending', '', '')
            ''', image_info_list)
        except Exception as e:
            log.error(f"Bulk insert failed: {e}")
            raise

    def add_or_update_image(self, image_id: str, image_path: str, mask_path: str, refined_mask_path: str) -> None:
        c = self.conn.cursor()
        try:
            c.execute(f'''
                INSERT OR IGNORE INTO {TABLE_NAME} (
                    image_id, image_path, mask_path, refined_mask_path, 
                    initial_tag, final_tag, tags, status, reviewer, timestamp
                )
                VALUES (?, ?, ?, ?, '', '', '', 'pending', '', '')
            ''', (image_id, image_path, mask_path, refined_mask_path))
        except Exception as e:
            log.error(f"Insert failed for {image_id}: {e}")

    def get_images_for_review(self, only_tags: List[str] = None) -> List[Tuple]:
        c = self.conn.cursor()
        try:
            where_clauses = []
            params = []

            if only_tags:
                tag_clauses = []
                for tag in only_tags:
                    tag_clauses.append("initial_tag LIKE ?")
                    params.append(f"%{tag}%")
                where_clauses.append("(" + " OR ".join(tag_clauses) + ")")

            base_query = f"SELECT * FROM {TABLE_NAME}"
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)

            c.execute(base_query, params)
            rows = c.fetchall()
            return rows
        except Exception as e:
            log.error(f"Error fetching images from DB: {e}")
            return []

    def bulk_update_tags(self, updates: list) -> None:
        """
        updates: list of tuples:
        (initial_tag, final_tag, tags, status, reviewer, timestamp, image_id)
        """
        log.info(f"Bulk updating tags for {len(updates)} images")
        c = self.conn.cursor()
        try:
            c.executemany(
                f"UPDATE {TABLE_NAME} SET initial_tag=?, final_tag=?, tags=?, status=?, reviewer=?, timestamp=?, refine_params=? WHERE image_id=?",
                updates
            )
        except Exception as e:
            log.error(f"Bulk tag update failed: {e}")
            raise

    def get_all_image_ids(self) -> Set[str]:
        c = self.conn.cursor()
        try:
            c.execute(f"SELECT image_id FROM {TABLE_NAME}")
            return set(row[0] for row in c.fetchall())
        except Exception as e:
            log.error(f"Error fetching image IDs from DB: {e}")
            return set()

    def get_images_for_refinement(self, only_tags: List[str] = None) -> List[Tuple]:
        # Only return those with a non-empty initial_tag and not already final_tag == "good"
        c = self.conn.cursor()
        try:
            base_query = f'''
                SELECT *
                FROM {TABLE_NAME}
                WHERE 
                    (final_tag IS NULL OR final_tag != "good")
                    AND (initial_tag IS NOT NULL AND TRIM(initial_tag) != "")
            '''
            params = []
            if only_tags:
                # Compose a WHERE clause for tags using LIKE, for safety and flexibility
                tag_clauses = []
                for tag in only_tags:
                    tag_clauses.append("initial_tag LIKE ?")
                    params.append(f"%{tag}%")
                tag_filter = " AND (" + " OR ".join(tag_clauses) + ")"
                base_query += tag_filter
            c.execute(base_query, params)
            rows = c.fetchall()
            return rows
        except Exception as e:
            log.error(f"Error fetching images for refinement: {e}")
            return []
    
    def close(self):
        try:
            self.conn.close()
        except Exception as e:
            log.error(f"Error closing DB: {e}")

    def __enter__(self):
        return self
    
    def commit(self):
        log.info("Committing DB transaction.")
        try:
            self.conn.commit()
        except Exception as e:
            log.error(f"DB commit failed: {e}")
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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
        self.refined_masks_dir = Path(cfg.paths.refined_masks_dir)
        self.db_path = cfg.paths.agir_field_db

        # Use persistent DB connection
        self.db = InspectionDB(self.db_path)
        self._populate_db_with_images()

        # Configuration parameters for FiftyOne Voxel
        self.port = cfg.mask_gen.inspect.port
        self.dataset_name = cfg.mask_gen.inspect.dataset_name
        self.dataset = None # Place holder for FiftyOne dataset
        self.session = None # Place holder for FiftyOne session

        # Canonical tag mapping for user-defined tags
        self.canonical_mapping = cfg.mask_gen.canonical_tag_mapping

        # Reviewer name for tagging
        self.reviewer = os.getenv("USER")

        self.only_tags = cfg.mask_gen.inspect.only_tags

    def _populate_db_with_images(self) -> None:
        
        try:
            log.info("Populating DB with missing images...")
            existing_ids = self.db.get_all_image_ids()
            masks_to_add = []
            for mask_path in self.mask_gen_cutout_dir.glob("*_mask.png"):
                image_name = mask_path.name.replace("_mask.png", ".jpg")
                if image_name in existing_ids:
                    log.debug(f"Skipping {image_name} (already in DB)")
                    continue
                image_path = str(self.mask_gen_cutout_dir / image_name)
                mask_path_str = str(mask_path)
                refined_path = str(self.refined_masks_dir / mask_path.name) if self.refined_masks_dir else ""
                masks_to_add.append((image_name, image_path, mask_path_str, refined_path))
            if 0 < len(masks_to_add) <= 3:
                log.info(f"Adding {len(masks_to_add)} images one by one to DB")
                for entry in masks_to_add:
                    try:
                        self.db.add_or_update_image(*entry)
                    except Exception as e:
                        log.error(f"Error adding image {entry[0]}: {e}")
            elif masks_to_add:
                log.info(f"Adding {len(masks_to_add)} images in bulk to DB")
                self.db.add_images_bulk(masks_to_add)
            else:
                log.info("No new images to add to the database.")
        except Exception as e:
            log.error(f"Error populating DB with images: {e}")

    def normalize_tags(self, user_tags: List[str]) -> List[str]:
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
        log.debug(f"Normalized tags {user_tags} -> {list(normalized)}")
        return list(normalized)

    def _load_samples(self) -> List[fo.Sample]:
        """
        Loads FiftyOne samples from the database.
        Returns:
            list: A list of FiftyOne samples created from the database entries.
        """
        db_rows = self.db.get_images_for_review(only_tags=self.only_tags)
        samples = []
        mask_paths_in_db = set()  # Track mask paths for deduplication

        for row in db_rows:
            (
                image_id, image_path, mask_path, refined_mask_path,
                initial_tag, final_tag, tags, status, reviewer, timestamp,
                refine_params_str
            ) = row

            mask_paths_in_db.add(str(mask_path))  # Store as string for easy comparison

            # Skip if files are missing
            if not Path(mask_path).exists():
                log.warning(f"Mask file not found for {image_path}. Skipping.")
                continue

            # Load the initial mask and create a sample
            initial_mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            sample = fo.Sample(filepath=image_path)
            sample["initial_mask"] = fo.Segmentation(mask=initial_mask_array)

            # Attach refined mask if present
            if refined_mask_path and Path(refined_mask_path).exists():
                try:
                    refined_mask_array = np.array(Image.open(refined_mask_path).convert("L"), dtype=np.uint8)
                    sample["refined_mask"] = fo.Segmentation(mask=refined_mask_array)
                except Exception as e:
                    log.warning(f"Error loading refined mask for {refined_mask_path}: {e}")

            # Always display the final tag if it exists and is non-empty
            display_tags = []
            
            if final_tag:
                display_tags = [t.strip() for t in final_tag.split(",") if t.strip()]
            elif initial_tag:
                display_tags = [t.strip() for t in initial_tag.split(",") if t.strip()]

            if display_tags:
                sample.tags = display_tags

            samples.append(sample)
        
        log.info(f"Loaded {len(samples)} samples (DB + new files).")
        return samples


    def _create_dataset(self, samples: List[fo.Sample]) -> fo.Dataset:
        """
        Creates a FiftyOne dataset from samples defined in the SQLite DB.
        If dataset with same name exists, it will be deleted (optionally you can skip this for persistence).

        Args:
            samples (list): Ignored—samples are loaded from DB.

        Returns:
            fo.Dataset: The newly created FiftyOne dataset.
        """
        log.info(f"Creating dataset: {self.dataset_name}")
        try:
            if self.dataset_name in fo.list_datasets():
                log.info(f"Dataset '{self.dataset_name}' already exists. Load it.")
                # dataset = fo.load_dataset(self.dataset_name)
                # dataset.add_samples(samples)
                fo.delete_dataset(self.dataset_name)  # removes registry entry

            
            dataset = fo.Dataset(self.dataset_name)
            dataset.add_samples(samples)
            return dataset
        except Exception as e:
            log.error(f"Error creating/loading dataset {self.dataset_name}: {e}")
            raise

    def _get_voxel_dataset_map(self) -> Dict[str, str]:
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

    def run_voxel_inspection(self) -> None:
        """
        Runs the full inspection workflow:
        """
        log.info("Starting voxel inspection workflow.")
        samples: List[fo.Sample] = self._load_samples()
        self.dataset: fo.Dataset = self._create_dataset(samples)
        self.session = fo.launch_app(self.dataset, port=self.port)
        log.info("FiftyOne session started. Waiting for tagging to finish...")

        try:
            self.session.wait()
        except KeyboardInterrupt:
            print("\nSession manually interrupted by user.")
        except Exception as e:
            log.error(f"FiftyOne session error: {e}")
        finally:
            try:
                self.session.refresh()
                self.session.close()
                log.info("FiftyOne session ended. Saving tags to database.")
            except Exception as e:
                log.warning(f"Error closing FiftyOne session: {e}")

        self._update_sqlite_db_with_tags()
        log.info("Voxel inspection workflow complete.")
        


    def _update_sqlite_db_with_tags(self) -> None:
        """
        After the session, updates SQLite DB with tags for all samples in the dataset.
        """
        timestamp = datetime.datetime.now().isoformat()
        updates = []
        for sample in self.dataset:
            image_name = Path(sample.filepath).name
            canonical_tags = self.normalize_tags(sample.tags)
            tags = set([t.lower() for t in canonical_tags if t])
            refine_params = None
            initial_tag = None
            final_tag = None
            status = None
            reviewer = None
            if "good" in tags:
                final_tag = "good"
                status = "reviewed"
                reviewer = self.reviewer
            elif tags:
                initial_tag = ",".join(sorted(tags))
                status = "pending"
                reviewer = self.reviewer
            else:
                status = "unreviewed"
                reviewer = None\
                
            if isinstance(tags, set):
                tags = ",".join(sorted(tags))
            updates.append((initial_tag, final_tag, tags, status, reviewer, timestamp, refine_params, image_name))
        self.db.bulk_update_tags(updates)
        self.db.commit()

    def __del__(self):
        try:
            self.db.close()
        except Exception as e:
            log.warning(f"Error on DB close: {e}")

def main(cfg: DictConfig) -> None:
    """
    Entry point for launching the voxel inspection process.
    """
    # TODO: Move images marked as good to long-term storage or test directory.
    log.info("Mask inspection script started.")
    try:
        inspector = FiftyOneMaskInspector(cfg)
        inspector.run_voxel_inspection()
    except Exception as e:
        log.error(f"Fatal error in main(): {e}")
    log.info("Mask inspection script finished.")