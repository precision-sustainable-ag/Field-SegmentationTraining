import getpass
import logging
import sys 
import hydra 
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

sys.path.append("src")

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the post-segmentation processing pipeline.
    
    Args:
        cfg (DictConfig): Configuration object containing paths and parameters.
    
    Raises:
        Exception: If any task fails during processing.
    """
    cfg = OmegaConf.create(cfg)
    whoami = getpass.getuser()

    tasks = cfg.pipeline
    log.info(f"Running {' ,'.join(tasks)} as {whoami}")

    for task in tasks:
        cfg.general.task = task
        try:
            task = get_method(f"{task}.main")
            task(cfg)
        except Exception as e:
            log.exception("Error in task %s: %s", task, e)
            sys.exit(1)

if __name__ == "__main__":
    main()