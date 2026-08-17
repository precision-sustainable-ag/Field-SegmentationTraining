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
    weights_path: str = cfg.inference.weights_path if 'weights_path' in cfg.inference else cfg.paths.yolo_weed_detection_model  
      
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
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        image_files.extend(glob.glob(os.path.join(input_source, ext)))
        
    if not image_files:
        print(f"No images found in {input_source}. Exiting inference.")
        return

    print(f"Found {len(image_files)} images for inference. Commencing prediction...")

    # 4. Extract YOLO26 / YOLOv8 Specific Parameters
    end2end_flag: bool = getattr(cfg.inference, "end2end", True)
    max_det_val: int = getattr(cfg.inference, "max_det", 300)
    conf_val: float = getattr(cfg.inference, "conf", 0.25)
    iou_val: float = getattr(cfg.inference, "iou", 0.45)
    # Map the new precision argument (fallback to 16 if missing)
    quantize_val: int = getattr(cfg.inference, "precision", 16)

    image_size: int = cfg.inference.image_processing.size.height if 'image_processing' in cfg.inference else 1024
    output_dir: str = cfg.paths.project_inference_dir
    # Natively pull Date and Time from Hydra to guarantee chronological sorting
    run_subfolder: str = f"detect_infer_{cfg.job.job_now_date}_{cfg.job.job_now_time}"
    # 5. Execute Prediction
    results = model.predict(
        source=input_source,
        device=device_arg,
        conf=conf_val,
        iou=iou_val,
        imgsz=image_size,
        quantize=quantize_val,
        max_det=max_det_val,
        end2end=end2end_flag,
        save=True,          
        save_txt=True,      
        save_conf=True,     
        project=output_dir, 
        name=run_subfolder
    )
    
    print(f"YOLO inference completed. Results saved to: {output_dir}")