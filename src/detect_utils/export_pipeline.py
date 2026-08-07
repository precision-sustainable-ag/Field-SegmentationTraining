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

    # 2. Extract Export Configuration Settings
    export_cfg = getattr(cfg.inference, "export", {})
    export_format: str = export_cfg.get("format", "engine")
    end2end_flag: bool = export_cfg.get("end2end", True)
    max_det_val: int = export_cfg.get("max_det", 300)
    quantize_val: int = export_cfg.get("quantize", 16)
    target_device: int = export_cfg.get("device", 0)
    workspace_gb: int = export_cfg.get("workspace", 4)

    log.info(f"Commencing export to {export_format.upper()} (end2end={end2end_flag}, quantize={quantize_val})...")
    
    try:
        exported_path: str = model.export(
            format=export_format,
            quantize=quantize_val,
            end2end=end2end_flag,
            max_det=max_det_val,
            imgsz=cfg.preprocess.image_processing.size.height,
            device=target_device,
            workspace=workspace_gb
        )
        log.info(f"YOLO export completed successfully. Engine saved to: {exported_path}")
        
    except Exception as e:
        log.error(
            f"Export failed. If exporting to TensorRT on an edge device, ensure TensorRT and "
            f"JetPack are correctly configured. Details: {e}"
        )