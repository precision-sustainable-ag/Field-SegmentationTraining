# src/detect_utils/export_pipeline.py

import os
import logging
from omegaconf import DictConfig
from ultralytics import YOLO

log = logging.getLogger(__name__)

def run_yolo_export(cfg: DictConfig) -> None:
    """
    Executes the YOLO export pipeline to compile the trained PyTorch model 
    into a highly optimized format (TensorRT by default) for edge deployment.
    
    Args:
        cfg (DictConfig): The global Hydra configuration object.
    """
    log.info("Initializing YOLO Export Pipeline...")

    # 1. Locate the weights
    # Point this to your newly trained weights, or the default path in config
    weights_path: str = cfg.paths.yolo_weed_detection_model
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[-] Model weights not found at: {weights_path}")
        
    log.info(f"Loading model weights from: {weights_path}")
    model = YOLO(weights_path, task='detect')

    # 2. Execute Export
    # Compiling a TensorRT engine requires the target image size and must be run on the GPU.
    log.info("Commencing export to TensorRT engine (FP16)...")
    
    try:
        exported_path = model.export(
            format="engine",       # TensorRT format for edge architectures
            half=True,             # FP16 precision for accelerated inference
            imgsz=cfg.preprocess.image_processing.size.height,
            device=0,              # TensorRT compilation must happen on a GPU
            workspace=4            # Allocates 4GB max workspace for the TRT builder
        )
        log.info(f"YOLO export completed successfully. Engine saved to: {exported_path}")
        
    except Exception as e:
        log.error(f"Export failed. Ensure you are running this on the target edge hardware with JetPack/TensorRT installed. Error: {e}")