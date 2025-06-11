import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class CustomDataset(Dataset):
    """
    A PyTorch Dataset class for loading images and their corresponding masks for segmentation tasks.
    """
    def __init__(self, root_path, test=False):
        """
        Initialize the dataset.

        Args:
            root_path (str): Directory containing train/test and corresponding masks.
            test (bool): If True, load test data. Otherwise, load training data.
        """
        self.aug_random_apply = transforms.RandomApply([
            transforms.ColorJitter(),
            transforms.RandomSolarize(threshold=int(200/255)),
        ], p=0.5)

        self.root_path = root_path
        if test:
            self.images = sorted([os.path.join(root_path, "test", i) for i in os.listdir(os.path.join(root_path, "test"))])
            self.masks = sorted([os.path.join(root_path, "test_masks", i) for i in os.listdir(os.path.join(root_path, "test_masks"))])
        else:
            self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])
            self.masks = sorted([os.path.join(root_path, "train_masks", i) for i in os.listdir(os.path.join(root_path, "train_masks"))])

    def get_transforms_dict(self):
        """
        Create a dictionary of the photometric transforms applied in the dataset.

        Returns:
            dict: A dictionary containing image transform configurations.
        """
        transforms_dict = {}
        for transform in self.aug_random_apply.transforms:
            transform_name = transform.__class__.__name__.lower()
            transform_details = {}

            for attr in dir(transform):
                if not attr.startswith("_") and not callable(getattr(transform, attr)):
                    if attr not in ["T_destination", "call_super_init", "dump_patches", "training", "threshold"]:
                        transform_details[attr] = getattr(transform, attr)

            transform_details["state"] = True
            transforms_dict[transform_name] = transform_details

        return transforms_dict

    def __getitem__(self, index):
        """
        Get image and corresponding mask with augmentations.

        Args:
            index (int): Index of data item.

        Returns:
            Tuple[Tensor, Tensor]: Transformed image and mask tensors.
        """
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        # Resize
        img = TF.resize(img, (512, 512))
        mask = TF.resize(mask, (512, 512))

        # Geometric augmentations (synchronized)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        # Photometric augmentations (only image)
        img = self.aug_random_apply(img)

        # Convert to tensors
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        mask = torch.round(mask)  # Ensure binary

        return img, mask

    def __len__(self):
        """
        Total number of items in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.images)

    def get_file_paths(self, idx):
        """
        Get file paths of image and mask.

        Args:
            idx (int): Index of data item.

        Returns:
            Tuple[str, str]: Image and mask file paths.
        """
        return self.images[idx], self.masks[idx]