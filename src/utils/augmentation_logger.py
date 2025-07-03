# src/utils/augmentation_logger.py

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from torchvision.utils import make_grid, save_image
import wandb


def log_augmentation_batch(train_loader, logger_cfgs, num_samples: int = 4):
    """
    One-off: log a single batch of augmented inputs and their corresponding masks.
    - Saves a grid of paired images & masks from `train_loader`
      to <hydra_run_dir>/image_logs/aug_inputs.png.
    - Logs that grid to any instantiated WandB logger.

    Args:
        train_loader: torch.utils.data.DataLoader
        logger_cfgs: list of OmegaConf logger configs (e.g., cfg.train.logger)
        num_samples: maximum number of pairs to include in the grid
    """
    # Only run on global-rank 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    if rank != 0:
        return

    # Grab one batch (image, mask)
    try:
        x, y = next(iter(train_loader))
    except StopIteration:
        return

    # Determine how many samples we have
    n = min(num_samples, x.size(0))
    imgs = []
    x_cpu = x[:n].cpu()
    y_cpu = y[:n].cpu()

    # Ensure mask shape [B, H, W]
    if y_cpu.dim() == 4 and y_cpu.size(1) == 1:
        y_cpu = y_cpu.squeeze(1)

    # Convert masks to 3-channel for visualization
    y_rgb = y_cpu.unsqueeze(1).repeat(1, 3, 1, 1)

    # Build list: [img1, mask1, img2, mask2, ...]
    for i in range(n):
        imgs.append(x_cpu[i])
        imgs.append(y_rgb[i])

    # Make grid with 2 rows: first row inputs, second row masks
    grid = make_grid(imgs, nrow=n)

    # Prepare output path
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