import logging
import hydra
from omegaconf import DictConfig

# Import the task functions
# TODO: move these into a subdirectory src/mask_gen when incorporating into larger pipeline
from src.detect_weeds import main as detect_weeds
from src.unet_segment_weeds import main as unet_segment_weeds
from src.mask_inspect import main as mask_inspect
from src.mask_refine import main as mask_refine


log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "detect_weeds": detect_weeds,
    "unet_segment_weeds": unet_segment_weeds,
    "mask_inspect": mask_inspect,
    "mask_refine": mask_refine,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting preprocessing tasks...")
    
    task_dict = cfg.tasks # TODO: change this to cfg.mask_gen.tasks when incorporating into larger pipeline

    for task, enabled in task_dict.items():

        if enabled:
            log.info(f"Running task {task}")

            if task in TASK_REGISTRY:
                log.info(f"Running task {task}")
                TASK_REGISTRY[task](cfg)
            else:
                log.error(f"Task {task} not found in preprocessing task registry")
                return
        else:
            log.info(f"Skipping task {task}. Enabled: {enabled}")
    
    
    log.info("Mask generation complete.")

if __name__ == "__main__":
    main()