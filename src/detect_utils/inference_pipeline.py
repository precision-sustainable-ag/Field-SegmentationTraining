import os
import glob
from typing import List, Union
from omegaconf import DictConfig
from ultralytics import YOLO

from src.utils.gpu_utils import select_available_gpus


def run_yolo_inference(cfg: DictConfig) -> None:
    """
    Executes the YOLO inference pipeline using native Ultralytics features.
    
    This function dynamically allocates GPUs, loads the specified model weights, 
    and runs batch inference on the configured input directory, saving the 
    annotated images automatically.

    Args:
        cfg (DictConfig): The global Hydra configuration object.
    """
    print("Initializing YOLO Inference Pipeline...")

    # 1. GPU Allocation
    # Fetch excluded GPUs from inference config, defaulting to [0] if absent
    exclude_ids: List[int] = cfg.inference.gpu.exclude_gpu_ids if 'gpu' in cfg.inference else [0]
    
    # We only need 1 GPU for standard inference, unless specifically requested otherwise
    max_gpus: int = cfg.inference.gpu.max_gpus if 'gpu' in cfg.inference else 1
    
    chosen_gpus: List[int] = select_available_gpus(
        max_gpus=max_gpus, 
        exclude_ids=exclude_ids,
        verbose=True
    )
    
    # YOLO accepts a list of integers (e.g., [1]) or 'cpu'
    device_arg: Union[List[int], str] = chosen_gpus if len(chosen_gpus) > 0 else 'cpu'

    # 2. Determine Model Weights
    # For inference, you typically want to load the specifically trained weights
    # mapped in your paths config, rather than the base architecture (yolov8s.pt)
    weights_path: str = cfg.paths.yolo_weed_detection_model
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
        
    print(f"Loading model weights from: {weights_path}")
    model = YOLO(weights_path, task='detect')

    # 3. Determine Input Data
    # Point to the test set by default, or an explicitly defined input directory
    input_source: str = cfg.inference.input_dir if 'input_dir' in cfg.inference else cfg.paths.test_images_dir
    
    if not os.path.exists(input_source):
        raise FileNotFoundError(f"Inference input directory not found: {input_source}")

    # Ensure there are actually images to process to prevent cryptic YOLO errors
    image_files: List[str] = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(glob.glob(os.path.join(input_source, ext)))
        
    if not image_files:
        print(f"No images found in {input_source}. Exiting inference.")
        return

    print(f"Found {len(image_files)} images for inference. Commencing prediction...")

    # 4. Execute Native Inference
    # We map configurations directly to YOLO's native prediction kwargs
    output_dir: str = cfg.paths.project_inference_dir
    
    results = model.predict(
        source=input_source,
        device=device_arg,
        conf=cfg.inference.roi.conf if 'roi' in cfg.inference else 0.25,
        iou=cfg.inference.roi.iou if 'roi' in cfg.inference else 0.45,
        imgsz=cfg.preprocess.image_processing.size.height,
        half=cfg.inference.seg.amp.enable if 'seg' in cfg.inference else True,  # Mixed precision
        save=True,          # Natively saves images with drawn bounding boxes
        save_txt=True,      # Saves raw .txt coordinate outputs for downstream analysis
        save_conf=True,     # Includes confidence scores in the .txt files
        project=output_dir, # Base output directory
        name=f"detect_run_{cfg.job.job_now_time}" # Subfolder mapping
    )
    
    print(f"YOLO inference completed. Results saved to: {output_dir}")