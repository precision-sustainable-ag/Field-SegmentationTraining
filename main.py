# main.py

import logging, warnings, traceback, os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.pipeline_log import PipelineLogger
from src.train import train
from src.mask_gen import main as mask_gen

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "train": train,
    "mask_gen": mask_gen,
    # "train": train, # For when we incorporate training into the pipeline
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Hydra output dir: {HydraConfig.get().runtime.output_dir}")
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    
    pipe_logger = PipelineLogger(run_dir, cfg.mode)

    # hook warnings
    orig_showwarning = warnings.showwarning
    def _capture_warning(msg, cat, fn, ln, file=None, line=None):
        text = warnings.formatwarning(msg, cat, fn, ln, line)
        pipe_logger.add_warning(text)
        return orig_showwarning(msg, cat, fn, ln, file, line)
    warnings.showwarning = _capture_warning

    success = False
    try:
        if cfg.mode not in TASK_REGISTRY:
            raise ValueError(f"Unknown mode '{cfg.mode}'")
        TASK_REGISTRY[cfg.mode](cfg)
        success = True

    except Exception as e:
        log.error(f"Error in {cfg.mode}: {e}", exc_info=True)
        log.debug(traceback.format_exc())
        pipe_logger.add_error(e)

    finally:
        pipe_logger.finalize(success=success)

if __name__ == "__main__":
    main()