import os
from typing import List, Union
import wandb
from omegaconf import DictConfig
from ultralytics import YOLO

from src.utils.gpu_utils import select_available_gpus
from src.utils.seed import set_seed
from src.detect_utils.dataset_builder import split_and_prepare_dataset


def run_yolo_training(cfg: DictConfig) -> None:
    """
    Executes the YOLO training pipeline.
    
    This function handles reproducibility seeding, dynamic GPU allocation,
    dataset preparation, Weights & Biases initialization, and mapping Hydra 
    configurations directly into the Ultralytics training engine.

    Args:
        cfg (DictConfig): The global Hydra configuration object containing 
                          paths, model architectures, and training hyperparameters.
    """
    # 1. Enforce strict reproducibility across Python, NumPy, and PyTorch
    print(f"Setting global seed to: {cfg.train.seed}")
    set_seed(cfg.train.seed)
    
    # 2. GPU Allocation
    # Reads GPU selection configuration directly from cfg.train.gpu
    if hasattr(cfg.train, "gpu") and cfg.train.gpu.enable:
        exclude_ids: List[int] = list(cfg.train.gpu.exclude_gpu_ids)
        max_gpus: int = cfg.train.gpu.max_gpus
    else:
        exclude_ids: List[int] = [0]
        max_gpus: int = getattr(cfg.train, "num_gpus", 1)

    chosen_gpus: List[int] = select_available_gpus(
        max_gpus=max_gpus, 
        exclude_ids=exclude_ids,
        verbose=True
    )
    
    # YOLO accepts a list of integers (e.g., [1, 2, 3]) for DDP or 'cpu'
    device_arg: Union[List[int], str] = chosen_gpus if len(chosen_gpus) > 0 else 'cpu'
    
    # YOLO accepts a list of integers (e.g., [1, 2]) for multi-GPU Distributed Data Parallel (DDP)
    # If no GPUs are found, fallback to CPU
    device_arg: Union[List[int], str] = chosen_gpus if len(chosen_gpus) > 0 else 'cpu'

    # 3. Prepare the dataset and generate data.yaml
    print("Preparing dataset and generating YOLO configuration...")
    data_yaml_path: str = split_and_prepare_dataset(cfg)

    # 4. Initialize Weights & Biases (if enabled in config)
    # Ultralytics natively hooks into wandb if the run is initialized beforehand
    if cfg.train.logger.wandb.enable:
        print("Initializing Weights & Biases logger...")
        wandb.init(
            project=cfg.train.logger.wandb.project,
            entity=cfg.train.logger.wandb.entity,
            name=cfg.train.logger.wandb.run_name or None,
            config=dict(cfg)  # Log the full Hydra config for experiment tracking
        )

    # 5. Initialize the Ultralytics Model
    # Uses the predefined weights (e.g., yolov8s.pt) and explicitly sets the task to 'detect'
    print(f"Initializing YOLO architecture: {cfg.model.name}")
    model = YOLO(cfg.model.name, task=cfg.model.task)

    # 6. Execute Training
    # We map the relevant custom hyperparameters from Hydra into the YOLO engine
    print("Commencing YOLO training loop...")
    model.train(
        data=data_yaml_path,
        epochs=cfg.train.max_epochs,
        batch=cfg.train.batch_size,
        imgsz=cfg.preprocess.image_processing.size.height,
        workers=cfg.train.num_workers,
        device=device_arg,
        seed=cfg.train.seed,
        deterministic=cfg.train.deterministic,  # Combines with cuDNN deterministic settings
        box=cfg.train.box,
        cls=cfg.train.cls,
        dfl=cfg.train.dfl,
        project=cfg.project.name,
        name=f"detect_train_{cfg.job.job_now_time}"  # Organizes output folders dynamically
    )
    
    # 7. Cleanup
    if wandb.run is not None:
        wandb.finish()
    
    print("YOLO training pipeline completed successfully.")