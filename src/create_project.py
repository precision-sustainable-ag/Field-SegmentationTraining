import logging
import hydra
from omegaconf import DictConfig

from src.create_projects_utils.filter_by_species import GroupImagesBySpecies, CreateProject

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def create_project(cfg: DictConfig) -> None:
    """ Main entry point for creating a new project """
    log.info(f"Creating a project at {cfg.paths.project_dir}")
    g = GroupImagesBySpecies(cfg)
    sampled_species = g.get_sampled_images()

    create_project = CreateProject(cfg)
    create_project.copy_from_lts_to_local(sampled_species)


    #TODO: After users are done with their mask corrections, it needs to be saved to lts saved directory


if __name__ == "__main__":
    create_project()