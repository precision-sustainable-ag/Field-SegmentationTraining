import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    """
    A PyTorch Dataset class for loading images and their corresponding masks for segmentation tasks.

    Attributes:
        root_path (str): Root directory containing training and testing data.
        images (list): Sorted list of image file paths.
        masks (list): Sorted list of mask file paths.
        transform (callable): Transformation to be applied to the images and masks.
    """
    def __init__(self, root_path, test=False):
        """
        Initialize the dataset by specifying the root directory and whether it's for testing or training.

        Args:
            root_path (str): Root directory containing `train/`, `train_masks/`, `test/`, and `test_masks/`.
            test (bool, optional): Flag to indicate whether the dataset is for testing. Defaults to False (training).
        """
        self.root_path = root_path
        if test:
            self.images = sorted([root_path + "/test/" + i for i in os.listdir(root_path + "/test/")])
            self.masks = sorted([root_path + "/test_masks/" + i for i in os.listdir(root_path + "/test_masks/")])
        else:
            self.images = sorted([root_path + "/train/" + i for i in os.listdir(root_path + "/train/")])
            self.masks = sorted([root_path + "/train_masks/" + i for i in os.listdir(root_path + "/train_masks/")])

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize images and masks to 512x512 pixels.
            transforms.ToTensor()          # Convert images and masks to PyTorch tensors.
        ])

    def __getitem__(self, index):
        """
        Get the image and mask at the specified index.

        Args:
            index (int): Index of the dataset item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and mask tensors.
        """
        img = Image.open(self.images[index]).convert("RGB")  # Open and convert image to RGB.
        mask = Image.open(self.masks[index]).convert("L")    # Open and convert mask to grayscale.

        return self.transform(img), self.transform(mask)

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.images)

    def get_file_paths(self, idx):
        """
        Get the file paths for the image and mask at the specified index.

        Args:
            idx (int): Index of the dataset item.

        Returns:
            Tuple[str, str]: File paths of the image and corresponding mask.
        """
        return self.images[idx], self.masks[idx]