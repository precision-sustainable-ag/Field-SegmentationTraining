import logging
from pathlib import Path
import pandas as pd
import sqlite3
import shutil

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from typing import Dict, Tuple, Any

log = logging.getLogger(__name__)

class CreateProject:
    """ Takes the cfg.create_project object and the filtered images by species name
        and creates a project directory structure and downloads developed images into that 
        project directory structure.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.local_developed_images_dir = Path(cfg.paths.project_maskgen_dir) / "developed-images"
        self.lts_source_dir = Path(cfg.paths.longterm_storage)
        self.local_developed_images_dir.mkdir(parents=True, exist_ok=True)
        self.field_image_batches_dir = Path(cfg.paths.field_batches_dir)  





    # fix this



    def copy_from_lts_to_local(self, sample_df: pd.DataFrame) -> None:
        for _, row in sample_df.iterrows():
            dst_dir = self.local_developed_images_dir
            developed_image_path = row["developed_image_path"]
            source_path = self.lts_source_dir / developed_image_path
            if source_path.exists():
                log.info(f"Copying {developed_image_path.name} from {source_path} to {dst_dir}")
                shutil.copy(source_path, dst_dir)
            else:
                log.warning(f"Source image {developed_image_path.name} does not exist at {source_path}. Skipping copy.")

class GroupImagesBySpecies:
    """ Read the sql db and filters images by species name and images per species
    to output a list of images to be downloaded.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.db_path = Path(cfg.paths.agir_field_db)
        self.species_of_interest = cfg.create_project.species
        self.num_images_per_species = cfg.create_project.images_per_species
        self.df_field_data = None

    def _load_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        self.df_field_data = pd.read_sql_query("SELECT * FROM field_data", conn)
        conn.close()
        return self.df_field_data

    def _filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter for jpg images
        filtered_df = df[df['extension'] == 'jpg']
        # Further filter for species of interest
        filtered_df = filtered_df[filtered_df['app_species'].isin(self.species_of_interest)]
        return filtered_df

    def _group_images_by_species(self, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        # Loop through each species and select images
        species_group_dict = {}
        for species in self.species_of_interest:
            species_df = filtered_df[filtered_df['app_species'] == species]
            if not species_df.empty:
                # Randomly select the specified number of images
                selected_images = species_df.sample(n=self.num_images_per_species)
                species_group_dict[species] = selected_images['image_id'].tolist()

        return species_group_dict
    
    def main_process_filter_by_species(self) -> Dict[str, Any]:
        """ Main process to filter images by species and number of images per species.
        """
        # Load the database
        self.df_field_data = self._load_db()
        # Filter the DataFrame
        filtered_df = self._filter_df(self.df_field_data)
        # Group images by species of interest
        species_group_dict = self._group_images_by_species(filtered_df)

        return species_group_dict