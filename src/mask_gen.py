import logging
import hydra
from omegaconf import DictConfig

# Import the task functions
# TODO: move these into a subdirectory src/mask_gen when incorporating into larger pipeline
from src.mask_gen_utils.detect import main as detect
from src.mask_gen_utils.segment import main as segment
from src.mask_gen_utils.inspect import main as inspect
from src.mask_gen_utils.refine import main as refine
from src.mask_gen_utils.upload_cvat import main as upload_cvat
from src.mask_gen_utils.export_cvat import main as export_cvat

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "detect": detect,
    "segment": segment,
    "inspect": inspect,
    "refine": refine,
    "upload_cvat": upload_cvat,
    "export_cvat": export_cvat,  
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting mask gen tasks...")
    
    task_dict = cfg.tasks.mask_gen # TODO: change this to cfg.mask_gen.tasks when incorporating into larger pipeline
    
    for task, enabled in task_dict.items():

        if enabled:
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