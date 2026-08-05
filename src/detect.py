import logging
from omegaconf import DictConfig

from src.detect_utils.train_pipeline import run_yolo_training
from src.detect_utils.inference_pipeline import run_yolo_inference

log = logging.getLogger(__name__)


def detect_mode(cfg: DictConfig) -> None:
    """
    Primary dispatcher for the 'detect' mode.
    
    Reads the global Hydra configuration and routes the execution to the 
    appropriate YOLO pipeline (training, inference, or exporting) based on 
    the boolean flags set under cfg.tasks.detect.

    Args:
        cfg (DictConfig): The global Hydra configuration object.
    """
    log.info("Initializing Detection Mode...")

    # 1. Route to Training Pipeline
    if cfg.tasks.detect.train:
        log.info("Task: Detection Training enabled.")
        run_yolo_training(cfg)
        
    # 2. Route to Inference Pipeline
    elif cfg.tasks.detect.inference:
        log.info("Task: Detection Inference enabled.")
        run_yolo_inference(cfg)  
        
    # 3. Route to Export Pipeline (e.g., TensorRT compilation)
    elif cfg.tasks.detect.export:
        log.info("Task: Detection Model Export enabled.")
        run_yolo_export(cfg)   
        
    # 4. Fallback warning
    else:
        log.warning(
            "Detect mode was triggered, but no specific detect tasks "
            "(train, inference, export) were enabled in the configuration."
        )