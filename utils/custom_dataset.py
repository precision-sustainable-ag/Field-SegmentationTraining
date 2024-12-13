import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            self.images = sorted([root_path+"/test/"+i for i in os.listdir(root_path+"/test/")])
            self.masks = sorted([root_path+"/test_masks/"+i for i in os.listdir(root_path+"/test_masks/")])
        else:
            self.images = sorted([root_path+"/train/"+i for i in os.listdir(root_path+"/train/")])
            self.masks = sorted([root_path+"/train_masks/"+i for i in os.listdir(root_path+"/train_masks/")])
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
    
    def get_file_paths(self, idx):
        """
        Get the file paths for the image and mask at the specified index.

        Args:
            idx: Index of the dataset item.

        Returns:
            Tuple of (image_path, mask_path).
        """
        return self.image_paths[idx], self.mask_paths[idx]