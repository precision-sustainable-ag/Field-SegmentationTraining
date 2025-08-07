# src/train_utils/visualize_dataloader.py

import matplotlib.pyplot as plt
import torchvision
import torch
from omegaconf import DictConfig
import hydra
from hydra import initialize, compose
import sys
from pathlib import Path

# Add project root to sys.path so that `src` becomes importable
sys.path.append(str(Path(__file__).resolve().parents[2]))
conf_dir = str(Path(__file__).resolve().parents[2] / "conf")

from src.train_utils.data.dataset import FieldDataset
from src.train_utils.data.collate import get_batch_collate_fn

@hydra.main(version_base="1.1", config_path=conf_dir, config_name="config")
def visualize(cfg: DictConfig):
    """
    Visualize a single batch from the training DataLoader.
    """
    # Instantiate dataset and loader
    ds = FieldDataset(cfg, mode="train")
    collate = get_batch_collate_fn(cfg.augment.train.batch)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=min(4, len(ds)),
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate
    )

    # Grab one batch
    images, masks = next(iter(loader))  # images: [B,3,H,W], masks: [B,1,H,W]

    # Create grid of images
    img_grid = torchvision.utils.make_grid(images, nrow=4, padding=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title("Batch Images")
    plt.axis("off")
    plt.savefig("batch_visualization_image.png")

    # Create grid of masks (expand to 3 channels for visualization)
    mask_grid = torchvision.utils.make_grid(masks.expand(-1, 3, -1, -1), nrow=4, padding=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_grid.permute(1, 2, 0))
    plt.title("Batch Masks")
    plt.axis("off")
    plt.savefig("batch_visualization_mask.png")

if __name__ == "__main__":
    visualize()

