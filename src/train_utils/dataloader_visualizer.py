# src/train_utils/visualize_dataloader.py

import math
import matplotlib.pyplot as plt
import torchvision
import torch
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import sys
from pathlib import Path

# Make 'src' importable
sys.path.append(str(Path(__file__).resolve().parents[2]))
conf_dir = str(Path(__file__).resolve().parents[2] / "conf")

from src.train_utils.data.dataset import FieldDataset
from src.train_utils.data.collate import get_batch_collate_fn

def _get_run_dir() -> Path:
    # Always prefer Hydra's run/output dir if initialized; otherwise fall back to CWD
    if HydraConfig.initialized():
        # In Hydra 1.3, this is the canonical per-run directory
        return Path(HydraConfig.get().runtime.output_dir)
    return Path.cwd()

@hydra.main(version_base="1.3", config_path=conf_dir, config_name="config")
def vis_dataloader_batch(cfg: DictConfig) -> None:
    """
    Visualize a single batch from the training DataLoader.
    Saves PNGs under the Hydra run directory: <.../outputs/.../image_logs/>.
    """
    # Dataset & loader
    ds = FieldDataset(cfg, mode="train")
    num_samples = int(cfg.train.dataloader_visualizer.num_samples)
    collate = get_batch_collate_fn(cfg.augment.train.batch)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=min(num_samples, len(ds)),
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    images, masks = next(iter(loader))  # images [B,3,H,W], masks [B,1,H,W] (float or long)

    # Figure out grid columns: try to be square-ish
    B = images.size(0)
    nrow = min(4, B) if B <= 8 else math.ceil(math.sqrt(B))

    # Prepare output dir under the Hydra run dir
    out_dir = _get_run_dir() / "image_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Images grid
    img_grid = torchvision.utils.make_grid(images, nrow=nrow, padding=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title("Batch Images")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "batch_visualization_image.png")
    plt.close()

    # Masks grid (expand to 3 channels for visualization)
    if masks.ndim == 4 and masks.size(1) == 1:
        masks_vis = masks.expand(-1, 3, -1, -1)
    elif masks.ndim == 3:  # [B,H,W] -> [B,3,H,W]
        masks_vis = masks.unsqueeze(1).expand(-1, 3, -1, -1)
    else:
        masks_vis = masks  # assume already [B,3,H,W]

    mask_grid = torchvision.utils.make_grid(masks_vis, nrow=nrow, padding=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_grid.permute(1, 2, 0))
    plt.title("Batch Masks")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "batch_visualization_mask.png")
    plt.close()

if __name__ == "__main__":
    vis_dataloader_batch()
