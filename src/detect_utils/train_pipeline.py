# src/detect_utils/train_pipeline.py

import os
import shutil
from typing import List, Union

from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO, settings

from src.utils.gpu_utils import select_available_gpus
from src.utils.seed import set_seed
from src.detect_utils.dataset_builder import split_and_prepare_dataset


def run_yolo_training(cfg: DictConfig) -> None:
    """
    Executes the YOLO object detection training pipeline.
    
    This function handles:
    1. Global reproducibility seeding.
    2. Dynamic GPU allocation and DDP (Distributed Data Parallel) setup.
    3. Dataset splitting and YOLO configuration generation.
    4. DDP-safe Weights & Biases initialization via OS environment variables.
    5. Directory caging to prevent Ultralytics from downloading artifacts to the root path.
    6. Automatic copying of the best-trained weights to a static local directory for inference.

    Args:
        cfg (DictConfig): The global Hydra configuration object containing paths, 
                          model architectures, and training hyperparameters.

    Returns:
        None
    """
    # 1. Enforce strict reproducibility across Python, NumPy, and PyTorch
    print(f"Setting global seed to: {cfg.train.seed}")
    set_seed(cfg.train.seed)
    
    # 2. GPU Allocation
    # Read GPU selection configuration directly from cfg.train.gpu
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

    # 3. Prepare the dataset and generate data.yaml
    print("Preparing dataset and generating YOLO configuration...")
    data_yaml_path: str = split_and_prepare_dataset(cfg)

    # Define the unique run name used for both folder creation and W&B logging
    run_name: str = f"detect_train_{cfg.job.job_now_time}"

    # 4. Setup Project-Specific Directories
    # Ensures that W&B logs and downloaded pretrained weights stay inside the project folder
    project_wandb_dir: str = os.path.join(cfg.paths.project_dir, "wandb")
    project_weights_dir: str = os.path.join(cfg.paths.project_dir, "pretrained_weights")
    
    os.makedirs(project_wandb_dir, exist_ok=True)
    os.makedirs(project_weights_dir, exist_ok=True)

    # 5. Configure DDP-Safe Weights & Biases and Ultralytics Settings
    if cfg.train.logger.wandb.enable:
        print("Configuring Weights & Biases for DDP...")
        
        # Pass credentials via environment variables so DDP child processes inherit them natively
        os.environ["WANDB_PROJECT"] = cfg.train.logger.wandb.project
        if cfg.train.logger.wandb.entity:
            os.environ["WANDB_ENTITY"] = cfg.train.logger.wandb.entity
        os.environ["WANDB_NAME"] = run_name
        os.environ["WANDB_DIR"] = cfg.paths.project_dir  # W&B automatically appends '/wandb' to this
        
        # Force W&B on and redirect pretrained weight downloads internally
        settings.update({
            'wandb': True,
            'weights_dir': project_weights_dir
        })
    else:
        settings.update({
            'wandb': False,
            'weights_dir': project_weights_dir
        })

    # 6. Build Consolidated YOLO Training Arguments
    train_args: Dict[str, Any] = OmegaConf.to_container(cfg.train, resolve=True)
    augment_args: Dict[str, Any] = (
        OmegaConf.to_container(cfg.augment, resolve=True) if hasattr(cfg, "augment") else {}
    )

    # Merge training and augmentation parameters
    yolo_kwargs: Dict[str, Any] = {**train_args, **augment_args}

    # Filter out pipeline-only configuration keys
    for key in ["gpu", "logger"]:
        yolo_kwargs.pop(key, None)

    # Enforce mandatory execution overrides
    yolo_kwargs["data"] = data_yaml_path
    yolo_kwargs["imgsz"] = cfg.preprocess.image_processing.size.height
    yolo_kwargs["device"] = device_arg
    yolo_kwargs["project"] = cfg.paths.project_dir
    yolo_kwargs["name"] = run_name

    # 7. Cage Ultralytics to project weights directory during execution
    original_cwd: str = os.getcwd()
    
    try:
        os.chdir(project_weights_dir)
        
        target_model_path: str = os.path.join(project_weights_dir, cfg.model.name)
        model_to_load: str = target_model_path if os.path.exists(target_model_path) else cfg.model.name
        
        print(f"Initializing YOLO architecture: {cfg.model.name}")
        model = YOLO(model_to_load, task=cfg.model.task)

        print("Commencing YOLO training loop...")
        model.train(**yolo_kwargs)
        
    finally:
        os.chdir(original_cwd)
    
    # 8. Auto-Copy Best Weights to Static Location
    # Calculate exactly where YOLO just saved the best weights from this run
    trained_weights_path: str = os.path.join(cfg.paths.project_dir, run_name, "weights", "best.pt")
    
    # Calculate the static destination from the Hydra paths configuration
    static_model_path: str = cfg.paths.local_yolo_weed_detection_model
    os.makedirs(os.path.dirname(static_model_path), exist_ok=True)
    
    if os.path.exists(trained_weights_path):
        shutil.copy(trained_weights_path, static_model_path)
        print(f"Successfully copied latest best weights to: {static_model_path}")
    else:
        print(f"Warning: Expected to find trained weights at {trained_weights_path} but they were missing.")

    print("YOLO training pipeline completed successfully.")