import logging
import hydra
from omegaconf import DictConfig

# Import the task functions
# TODO: move these into a subdirectory src/mask_gen when incorporating into larger pipeline
# from src.finalize_lts_utils.finalize_lts import main as finalize_lts
from src.finalize_lts_utils.final_inspection import main as final_inspection
log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    # "finalize_lts": finalize_lts,
    "final_inspection": final_inspection,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting finalize tasks...")
    
    task_dict = cfg.tasks.finalize
    
    for task, enabled in task_dict.items():

        if enabled:
            if task in TASK_REGISTRY:
                log.info(f"Running task {task}")
                TASK_REGISTRY[task](cfg)
            else:
                log.error(f"Task {task} not found in finalize task registry")
                return
        else:
            log.info(f"Skipping task {task}. Enabled: {enabled}")


    log.info("Finalize tasks complete.")

if __name__ == "__main__":
    main()