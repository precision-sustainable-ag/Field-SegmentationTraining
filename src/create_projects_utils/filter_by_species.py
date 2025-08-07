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
    """ Takes the cfg.create_project object and the filtered images by species name
        and creates a project directory structure and downloads developed images into that 
        project directory structure.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.mask_gen_dir = Path(cfg.paths.project_maskgen_dir)
        self.local_developed_images_dir = self.mask_gen_dir / "developed-images"
        self.local_developed_images_dir.mkdir(parents=True, exist_ok=True)
        self.lts_source_dir = Path(cfg.paths.longterm_storage)

    def copy_from_lts_to_local(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        """ Parse the database to copy the developed images from the long-term storage to the local project directory while
        updating the sample_df with a new column for local image paths and species.
        
        Returns: Updated sample_df with local image paths and species.
        """
        sampled_df_copy = sample_df.copy()
        for _, row in sampled_df_copy.iterrows():
            dst_dir = self.local_developed_images_dir
            developed_image_path = row["developed_image_path"]
            source_path = self.lts_source_dir / developed_image_path
            if source_path.exists():
                # TODO: Update the DataFrame with the local path using the image_id index
                

                # Copy the image from LTS to local directory
                log.info(f"Copying {source_path.name} from {source_path} to {dst_dir}")
                shutil.copy(source_path, dst_dir)
            else:
                log.warning(f"Source image {developed_image_path.name} does not exist at {source_path}. Skipping copy.")
        
        return sample_df


    def save_temp_db(self, sampled_df: pd.DataFrame) -> None:
        """
        Save the sampled_df as a csv file that stores the sampled images, their species, and local image paths, 
        saving it in the project directory under mask_gen/cutouts/temp_db.csv.
        """
        temp_db_path = Path(self.mask_gen_dir) / "temp_db.csv"
        sampled_df.to_csv(temp_db_path, index=False)
        log.info(f"Temporary database saved at {temp_db_path}")

class GroupImagesBySpecies:
    """ Read the sql db and filters images by species name and images per species
    to output a list of images to be downloaded.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.db_path = Path(cfg.paths.agir_field_db)
        self.species_image_dict = cfg.create_project.species_images
        self.random_state = cfg.create_project.seed
        self.conn: Optional[sqlite3.Connection] = None

    def _load_db(self) -> None:
        df = pd.read_sql_query("SELECT * FROM field_data", self.conn)
        filtered_df = df[(df['extension'] == 'jpg') & (df['is_preprocessed'])]
        # Filter out first two images in the sample which are without mat or with color checker
        filtered_df = filtered_df[(filtered_df['image_index'] != 0) | (filtered_df['image_index'] != 1)]
        return filtered_df

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        log.info(f"Connected to {self.db_path}")

    def close(self):
        if self.conn:
            self.conn.close()
            log.info("Database connection closed.")

    def sample_by_n_species(self, df: pd.DataFrame) -> pd.DataFrame:
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
        """ Main process to filter images by species and number of images per species.
        """
        try:
            self.connect()
            # Load the database
            df = self._load_db()
        finally:
            self.close()
        # Filter the DataFrame
        return self.sample_by_n_species(df)