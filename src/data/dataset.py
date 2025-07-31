# src/data/dataset.py

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from omegaconf import OmegaConf, DictConfig
import json

# ─── import your augment builders ─────────────────────────────────────────────
from src.data.augment import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    get_noop_transform
)

class FieldDataset(Dataset):
    """
    Dataset for field segmentation tasks using image-mask pairs.

    - For 'train'/'val'/'test' modes, reads from preprocess directory.
    - Applies joint preprocessing + augmentation transforms.
    - Provides utility methods to inspect applied transforms and file paths.
    """

    def __init__(self, cfg: DictConfig, mode: str = "train") -> None:
        """
        Initialize the dataset by reading image and mask paths and setting up transforms.

        Args:
            cfg (DictConfig): Hydra configuration.
            mode (str): Dataset split to load: 'train', 'val', or 'test'.
        """
        # Keep references to the preprocess and augment config blocks
        self.cfg_pre = cfg.preprocess
        self.cfg_aug = cfg.augment

        # Determine image/mask directories based on mode
        if mode in ("train"):
            img_dir = Path(cfg.paths.train_images_dir)
            mask_dir = Path(cfg.paths.train_masks_dir)
        elif mode == "val":
            img_dir = Path(cfg.paths.val_images_dir)
            mask_dir = Path(cfg.paths.val_masks_dir)
        elif mode == "test":
            img_dir = Path(cfg.paths.test_images_dir)
            mask_dir = Path(cfg.paths.test_masks_dir)
        else:
            raise ValueError(f"Unsupported mode: {mode!r}. Choose from 'train','val','test'.")

        # List and sort all image and mask files
        self.images = sorted(img_dir.glob("*"))
        self.masks  = sorted(mask_dir.glob("*"))

        # Sanity check: ensure equal number of images and masks
        if len(self.images) != len(self.masks):
            raise RuntimeError(
                f"Number of images ({len(self.images)}) "
                f"and masks ({len(self.masks)}) do not match."
            )

        # ─── Dataset‐wide normalization setup ──────────────────────────────────
        # Use the flag in cfg.train to decide whether to normalize
        self.use_norm = bool(getattr(cfg.train, "use_data_normalization", False))
        if self.use_norm:
            # Stats JSON lives in paths.project_datastats_dir/rgb_mean_std.json
            stats_path = Path(cfg.paths.project_datastats_dir) / "rgb_mean_std.json"
            with open(stats_path, "r") as f:
                stats = json.load(f)
            # Create a torchvision Normalize transform
            self.normalize = transforms.Normalize(mean=stats["mean"], std=stats["std"])
        else:
            self.normalize = None
        
        # ─── build albumentations pipeline based on mode ───────────────────
        # If augmentations are enabled in config, build the appropriate transforms
        # Use the flag in cfg.train to decide whether to augment
        self.use_augment = bool(getattr(cfg.train, "use_data_augmentation", False))
        if self.use_augment:
            if mode == "train":
                self.transform = get_train_transforms(cfg)
            elif mode == "val":
                self.transform = get_val_transforms(cfg)
            else:
                self.transform = get_test_transforms(cfg)
        
        # no-op: return image & mask untouched
        else:
            self.transform = get_noop_transform()

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of image/mask pairs.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Fetch the image and mask at index `idx`, apply transforms, and return tensors.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image tensor shaped (C, H, W), dtype=torch.float32
                - mask tensor shaped (1, H, W), binary {0,1}, dtype=torch.float32
        """
        # Load PIL images
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")  # single channel mask

        # Convert to tensors in [0,1]
        # img_tensor = transforms.ToTensor()(img)
        # mask_tensor = transforms.ToTensor()(mask)

        # apply albumentations (numpy arrays in/out)
        arr = self.transform(
            image = np.array(img),
            mask = np.array(mask)
        )
        img_tensor = arr["image"].float() / 255.0  # convert to float32
        # mask is single channel, so we add a channel dimension and convert to float32
        mask_tensor = arr["mask"].unsqueeze(0).float()

        # Apply dataset-wide normalization if enabled
        if self.normalize is not None:
            img_tensor = self.normalize(img_tensor)

        # Ensure mask is binary: any value >0.5 becomes 1.0, else 0.0
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor

    def get_transforms_dict(self) -> Dict[str, Any]:
        """
        Return the active preprocess and augment configurations as plain dictionaries.

        Useful for logging or debugging which transforms were applied.

        Returns:
            dict: {
                "preprocess": <dict of preprocess settings>,
                "augment": <dict of augment settings>
            }
        """
        return {
            "preprocess": OmegaConf.to_container(self.cfg_pre, resolve=True),
            "augment":    OmegaConf.to_container(self.cfg_aug, resolve=True),
        }

    def get_file_paths(self, idx: int) -> Tuple[str, str]:
        """
        Retrieve the original file paths for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[str, str]: (image_path, mask_path)
        """
        return str(self.images[idx]), str(self.masks[idx])
