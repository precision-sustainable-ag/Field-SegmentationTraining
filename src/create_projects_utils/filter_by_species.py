import logging
from pathlib import Path

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
    def create_project_structure(self) -> None:
        """"""
    def download_images(self, images: Dict[str, Any]) -> None:
        """"""


class FilterBySpecies:
    """ Read the sql db and filters images by species name and images per species
    to output a list of images to be downloaded.
    """
    def project_grouping(self) -> Dict[str, Any]:
        """ Read the sql db and filters images by species name and images per species
            to output a list of images to be downloaded.
        """
        
    

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """"""

    filter_by_species = FilterBySpecies(cfg)
    species_group_dict = filter_by_species.project_grouping()

    create_project = CreateProject(cfg)
    create_project.create_project_structure()
    create_project.download_images(species_group_dict)

if __name__ == "__main__":
    main()