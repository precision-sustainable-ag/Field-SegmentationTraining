import logging
from pathlib import Path
import pandas as pd
import sqlite3
import shutil

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple, Any

log = logging.getLogger(__name__)

class CreateProject:
    def __init__(self, cfg: DictConfig) -> None:
        """ Initialize the CreateProject with Hydra configs."""
        self.mask_gen_dir = Path(cfg.paths.project_maskgen_dir)
        self.local_developed_images_dir = self.mask_gen_dir / "developed-images"
        self.local_developed_images_dir.mkdir(parents=True, exist_ok=True)
        self.lts_source_dir = Path(cfg.paths.longterm_storage)

    def copy_from_lts_to_local(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        """ Copy images from the long-term storage to the local project directory.
        Args:
            sample_df (pd.DataFrame): DataFrame containing the sampled images.
        Returns:
            pd.DataFrame: DataFrame with updated local image paths.
        """
        sampled_df_copy = sample_df.copy()
        for idx, row in sampled_df_copy.iterrows():
            dst_dir = self.local_developed_images_dir
            developed_image_path = row["developed_image_path"]
            source_path = self.lts_source_dir / developed_image_path
            if source_path.exists():    
                # Copy the image from LTS to local directory
                log.info(f"Copying {source_path} to {dst_dir}")
                shutil.copy(source_path, dst_dir)
                # Update the DataFrame with the local path
                sampled_df_copy.at[idx, "local_image_path"] = dst_dir / source_path.name
            else:
                log.warning(f"Source image {developed_image_path.name} does not exist at {source_path}. Skipping copy.")

        return sampled_df_copy

    def save_temp_db(self, sampled_df: pd.DataFrame) -> None:
        """ Save a temporary database in the project directory.
        Args:
            sampled_df (pd.DataFrame): DataFrame containing the sampled images.
        """
        temp_db_path = Path(self.mask_gen_dir) / "temp_db.csv"
        sampled_df.to_csv(temp_db_path, index=False)
        log.info(f"Temporary database saved at {temp_db_path}")

class FilterImagesBySpecies:
    """
    This class filters images from the database by species and number of images per species.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """ Initialize the FilterImagesBySpecies with Hydra configs."""
        self.db_path = Path(cfg.paths.agir_field_db)
        self.species_image_dict = cfg.create_project.species_images
        self.random_state = cfg.create_project.seed
        self.conn: Optional[sqlite3.Connection] = None

    def _load_db(self) -> None:
        """ Load the database and filter images based on extension and preprocessing status."""
        log.info(f"Loading the agir field db...")
        df = pd.read_sql_query("SELECT * FROM field_data", self.conn)
        filtered_df = df[(df['extension'] == 'jpg') & (df['is_preprocessed'])]
        # Filter out first two images in the sample which are without mat or with color checker
        filtered_df = filtered_df[(filtered_df['image_index'] != 0) | (filtered_df['image_index'] != 1)]
        return filtered_df

    def connect(self) -> None:
        """ Connect to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        log.info(f"Connected to agir field db: {self.db_path}")

    def close(self) -> None:
        """ Close the database connection."""
        if self.conn:
            self.conn.close()
            log.info("Database connection closed.")

    def sample_by_n_species(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample images from the DataFrame by species and number of images per species.
        Args:
            df (pd.DataFrame): DataFrame containing the images and their species.
        Returns:
            pd.DataFrame: DataFrame containing the sampled images.
        """
        log.info(f"Sampling images by species.")
        sampled_dfs = []
        # Loop through your config species_image_dict
        for species, n in self.species_image_dict.items():
            # Species must match the 'app_species' column in the DataFrame
            sub_df = df[df['app_species'] == species]
            if len(sub_df) == 0:
                log.warning(f"No images found for species: {species}")
                continue  # No rows for this species
            # If there are fewer rows than requested, sample all available
            sample_n = min(len(sub_df), n)
            sampled = sub_df.sample(n=sample_n, random_state=self.random_state )
            sampled_dfs.append(sampled)
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def get_sampled_images(self) -> pd.DataFrame:
        """
        Main process to filter images by species and number of images per species.
        Returns:
            pd.DataFrame: DataFrame containing the sampled images.
        """
        try:
            self.connect()
            # Load the database
            df = self._load_db()
        finally:
            self.close()
        # Filter the DataFrame
        return self.sample_by_n_species(df)