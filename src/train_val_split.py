import shutil
import random
import logging
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

def split_data(cfg: DictConfig, test_ratio=0.15, train_ratio=0.7, seed=42):
    """
    Splits the dataset into training, validation, and test sets based on specified ratios, 
    ensuring that each set contains paired image and mask files.

    Args:
        cfg (DictConfig): Configuration object that holds paths to the dataset directories 
            (e.g., `image_dir` for images, `yolo_format_label` for labels) and the project root.
        test_ratio (float, optional): Proportion of data to reserve for testing. Defaults to 0.15.
        train_ratio (float, optional): Proportion of the non-test data to reserve for training. 
            Defaults to 0.7 (70% training, 30% validation).
        seed (int, optional): Random seed for reproducibility of the splits. Defaults to 42.

    Steps:
        1. Loads the image and mask file paths from the directories specified in the configuration.
        2. Checks that the number of image files matches the number of mask files.
        3. Splits the data into train, validation, and test sets using `train_test_split`.
        4. Creates output directories for each split if they do not already exist.
        5. Copies the images and masks to their respective directories for each set.

    Raises:
        AssertionError: If the number of image files does not match the number of mask files, 
            ensuring each image has a corresponding mask.

    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define paths to the source image and mask directories
    cropped_resized_images_dir = Path(cfg.paths.cropped_resized_image_dir)
    yolo_cropped_resized_masks_dir = Path(cfg.paths.yolo_format_label)
    output_root_dir = Path(cfg.paths.data_dir, cfg.project_name)
    
    # Collect all mask file paths and generate corresponding image paths
    masks = sorted(yolo_cropped_resized_masks_dir.iterdir())
    mask_stems = [mask.stem for mask in masks]
    images = [Path(cropped_resized_images_dir, f"{stem}.jpg") for stem in mask_stems]
    
    # Ensure the number of images and masks match
    assert len(images) == len(masks), (
        f"Mismatch between images ({len(images)}) and masks ({len(masks)}). Each image must have a corresponding mask."
    )
    
    # Split data into training + validation and test sets
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        images, masks, test_size=test_ratio, random_state=seed
    )

    # Split remaining training + validation data into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=1 - train_ratio, random_state=seed
    )

    # Define directories for each subset (train, validation, test)
    train_images_dir = output_root_dir / 'images' / 'train'
    val_images_dir = output_root_dir / 'images' / 'val'
    test_images_dir = output_root_dir / 'images' / 'test'
    train_masks_dir = output_root_dir / 'labels' / 'train'
    val_masks_dir = output_root_dir / 'labels' / 'val'
    test_masks_dir = output_root_dir / 'labels' / 'test'
    
    # Create necessary directories
    for dir_path in [train_images_dir, val_images_dir, test_images_dir, train_masks_dir, val_masks_dir, test_masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Move the files to the respective directories
    for img in tqdm(train_images, desc="Copying training images"):
        shutil.copy2(img, train_images_dir / img.name)
    for img in tqdm(val_images, desc="Copying validation images"):
        shutil.copy2(img, val_images_dir / img.name)
    for img in tqdm(test_images, desc="Copying test images"):
        shutil.copy2(img, test_images_dir / img.name)

    for mask in tqdm(train_masks, desc="Copying training masks"):
        shutil.copy2(mask, train_masks_dir / mask.name)
    for mask in tqdm(val_masks, desc="Copying validation masks"):
        shutil.copy2(mask, val_masks_dir / mask.name)
    for mask in tqdm(test_masks, desc="Copying test masks"):
        shutil.copy2(mask, test_masks_dir / mask.name)
    
    log.info(
        f"Data split completed: {len(train_images)} training, {len(val_images)} validation, and {len(test_images)} test samples."
    )

def main(cfg: DictConfig) -> None:
    """
    Main function to initialize the dataset splitting process.

    Args:
        cfg (DictConfig): Configuration object containing paths and project settings.

    Logs:
        The beginning and completion of the dataset split process.
    """
    log.info("Starting dataset split into training, validation, and test sets.")
    split_data(cfg)
    log.info("Dataset split process completed.")
