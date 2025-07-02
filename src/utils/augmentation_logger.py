# src/utils/augmentation_logger.py

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from torchvision.utils import make_grid, save_image
import wandb


def log_augmentation_batch(train_loader, logger_cfgs, num_samples: int = 4):
    """
    One-off: log a single batch of augmented inputs.
    - Saves a grid of `num_samples` images from `train_loader`
      to <hydra_run_dir>/image_logs/aug_inputs.png.
    - Logs that image to any instantiated WandB or TensorBoard loggers.

    Args:
        train_loader: torch.utils.data.DataLoader
        logger_cfgs: list of OmegaConf logger configs (e.g., cfg.train.logger)
        num_samples: maximum number of images to include in the grid
    """
    # Only run on global-rank 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    if rank != 0:
        return

    # Grab one batch
    try:
        x, _ = next(iter(train_loader))
    except StopIteration:
        return

    # Build grid
    n = min(num_samples, x.size(0))
    grid = make_grid(x[:n].cpu(), nrow=n)

    # Prepare output dir
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    img_dir = run_dir / "image_logs"
    img_dir.mkdir(exist_ok=True, parents=True)
    aug_path = img_dir / "aug_inputs.png"
    save_image(grid, str(aug_path))
    print(f"Saved augmentation grid to {aug_path}")

    # Instantiate and log to each logger
    for lg_cfg in logger_cfgs:
        logger = hydra.utils.instantiate(lg_cfg)
        exp = getattr(logger, "experiment", None)
        if exp is None:
            continue
        # WandB
        if hasattr(exp, "log"):
            exp.log({"train/aug_inputs": [wandb.Image(str(aug_path))]})
        # TensorBoard
        elif hasattr(exp, "add_image"):
            exp.add_image("train/aug_inputs", grid, 0)
