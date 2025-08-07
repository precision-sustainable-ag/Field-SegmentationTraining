import logging
import hydra
from omegaconf import DictConfig
from src.create_projects_utils.filter_by_species import FilterImagesBySpecies, CreateProject

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def create_project(cfg: DictConfig) -> None:
    """ Main entry point for creating a new project """
    log.info(f"Creating a project at {cfg.paths.project_dir}")
    
    # Filter images by species and get a DataFrame of sampled images from db
    g = FilterImagesBySpecies(cfg)
    sampled_species = g.get_sampled_images()
    
    # Create the project directory structure, copy images, and save the temporary database
    create_project = CreateProject(cfg)
    sampled_df = create_project.copy_from_lts_to_local(sampled_species)

    # Save a temporary database in the project directory
    create_project.save_temp_db(sampled_df)
    log.info("Project creation complete.")

if __name__ == "__main__":
    create_project()