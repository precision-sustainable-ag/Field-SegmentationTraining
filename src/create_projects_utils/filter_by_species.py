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
    def __init__(self, cfg: DictConfig, species_group_dict: Dict[str, Any]) -> None:
        self.local_developed_images_dir = Path(cfg.paths.project_maskgen_dir) / "developed-images"
        self.local_developed_images_dir.mkdir(parents=True, exist_ok=True)
        self.field_image_batches_dir = Path(cfg.paths.field_batches_dir)  
        # Dict to hold species and their corresponding image IDs
        self.species_group_dict = species_group_dict





    # fix this



    
    def _get_path_for_image_id(self, image_id: str) -> Path | None:
        try:
            for subdir in self.field_image_batches_dir.iterdir():
                lts_developed_images_dir = subdir / "developed-images"
                for image_path in lts_developed_images_dir.glob("*.jpg"):
                    if image_path.name == image_id:
                        log.info(f"Found image {image_id} at {image_path}")
                return image_path
        except Exception as e:
            log.exception(f"Error occurred while getting path for image {image_id}: {e}")
    
    def copy_from_lts_to_local(self) -> None:
        for species, image_ids in self.species_group_dict.items():
            species_dir = self.local_developed_images_dir / species
            species_dir.mkdir(parents=True, exist_ok=True)
            for image_id in image_ids:
                source_path = self._get_path_for_image_id(image_id)
                log.info(f"Copying {image_id} from {source_path} to {species_dir}")
                # shutil.copy(source_path, species_dir / image_id)

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