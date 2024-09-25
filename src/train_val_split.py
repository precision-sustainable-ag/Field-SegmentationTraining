import shutil
import random
import logging
from pathlib import Path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

def split_data(cfg: DictConfig,  train_ratio=0.9, seed=42):
    random.seed(seed)
    
    images_dir = Path(cfg.paths.image_dir)
    masks_dir =  Path(cfg.paths.yolo_format_label)
    output_root_dir = Path(cfg.paths.data_dir, cfg.project_name)
    
    # Get list of all files in the images and masks directories
    masks = sorted(masks_dir.iterdir())
    mask_stems = [mask.stem for mask in masks]
    images = [Path(images_dir, f"{stem}.jpg") for stem in mask_stems]
    
    # Ensure the number of images and masks match
    assert len(images) == len(masks), f"The number of images ({len(images)}) and masks ({len(masks)}) should be the same"
    
    # Split the data using train_test_split
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, train_size=train_ratio, random_state=seed
    )
    
    # Create directories for train and validation sets
    train_images_dir = output_root_dir / 'images' / 'train'
    val_images_dir = output_root_dir / 'images' / 'val'
    train_masks_dir = output_root_dir / 'labels' / 'train'
    val_masks_dir = output_root_dir / 'labels' / 'val'
    
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_masks_dir.mkdir(parents=True, exist_ok=True)
    val_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Move the files to the respective directories
    for img in train_images:
        shutil.copy2(img, train_images_dir / img.name)
    for img in val_images:
        shutil.copy2(img, val_images_dir / img.name)
    for mask in train_masks:
        shutil.copy2(mask, train_masks_dir / mask.name)
    for mask in val_masks:
        shutil.copy2(mask, val_masks_dir / mask.name)
    
    print(f"Data split completed. {len(train_images)} training and {len(val_images)} validation samples.")

def main(cfg: DictConfig) -> None:
    
    split_data(cfg)
