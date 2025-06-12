# src/data/dataset.py

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

class FieldDataset(Dataset):
    """
    A Dataset for field segmentation tasks.

    - For 'train'/'val' modes, reads from processed directories.
    - For 'test' mode, reads from raw test directories.
    - Applies joint preprocessing + augmentation transforms.
    - Provides utility methods to inspect applied transforms and file paths.
    """

    def __init__(self, cfg, mode: str = "train"):
        """
        Initialize the dataset.

        Args:
            cfg: Hydra config object.
            mode: One of "train", "val", or "test". Determines which folders to read.
        """
        # Keep references to the preprocess and augment config blocks
        self.cfg_pre = cfg.preprocess
        self.cfg_aug = cfg.augment

        # Determine image/mask directories based on mode
        if mode in ("train", "val"):
            img_dir = Path(cfg.preprocess.processed_image_dir)
            mask_dir = Path(cfg.preprocess.processed_mask_dir)
        elif mode == "test":
            img_dir = Path(cfg.paths.test_image_dir)
            mask_dir = Path(cfg.paths.test_mask_dir)
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

        # Build the joint transform function (returns img_tensor, mask_tensor)
        from data.augmentation import build_transforms
        self.transform = build_transforms(self.cfg_pre, self.cfg_aug)

    def __len__(self):
        """
        Returns:
            int: Total number of image/mask pairs.
        """
        return len(self.images)

    def __getitem__(self, idx):
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

        # Apply the combined preprocessing + augmentation transforms
        img_tensor, mask_tensor = self.transform(img, mask)

        # Ensure mask is binary: any value >0.5 becomes 1.0, else 0.0
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor

    def get_transforms_dict(self):
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

    def get_file_paths(self, idx):
        """
        Retrieve the original file paths for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[str, str]: (image_path, mask_path)
        """
        return str(self.images[idx]), str(self.masks[idx])
