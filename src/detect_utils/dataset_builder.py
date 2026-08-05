# src/detect_utils/dataset_builder.py

import os
import glob
import shutil
import random
import yaml
from typing import List, Tuple, Dict
from omegaconf import DictConfig


def create_dirs(dirs_list: List[str]) -> None:
    """
    Safely creates a list of directories.

    Args:
        dirs_list (List[str]): A list of absolute or relative directory paths to create.
    """
    for d in dirs_list:
        os.makedirs(d, exist_ok=True)


def build_yolo_yaml(cfg: DictConfig) -> str:
    """Generates the Ultralytics data.yaml file by reading classes.txt."""
    yaml_save_path: str = os.path.join(cfg.paths.project_train_dir, "data.yaml")
    os.makedirs(cfg.paths.project_train_dir, exist_ok=True)

    # Read class names directly from classes.txt in the raw labels folder
    classes_file = os.path.join(cfg.paths.raw_detect_labels_dir, "classes.txt")
    class_names = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Fallback just in case
    if not class_names:
        class_names = ["target_weed"]

    # Construct the YAML dictionary
    yolo_data: Dict[str, any] = {
        'path': cfg.paths.preprocess_split_dir,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'test': os.path.join('test', 'images') if cfg.preprocess.split.test > 0 else '',
        'names': {i: name for i, name in enumerate(class_names)}
    }

    # Write to disk
    with open(yaml_save_path, 'w') as f:
        yaml.dump(yolo_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"YOLO data.yaml generated at: {yaml_save_path}")
    print(f"Classes mapped: {yolo_data['names']}")
    return yaml_save_path

def split_and_prepare_dataset(cfg: DictConfig) -> str:
    """
    Shuffles and splits raw images and labels into train/val/test sets,
    copies them to the destination directories, and generates the YOLO data.yaml.

    Args:
        cfg (DictConfig): The global Hydra configuration object.

    Returns:
        str: The absolute path to the generated data.yaml file.
    """
    # 1. Map paths from the Hydra config
    raw_images_dir: str = cfg.paths.raw_detect_images_dir
    raw_labels_dir: str = cfg.paths.raw_detect_labels_dir
    
    # Dictionary mapping split names to a tuple of (image_dest_dir, label_dest_dir)
    dirs: Dict[str, Tuple[str, str]] = {
        'train': (cfg.paths.train_images_dir, cfg.paths.train_labels_dir),
        'val': (cfg.paths.val_images_dir, cfg.paths.val_labels_dir),
        'test': (cfg.paths.test_images_dir, cfg.paths.test_labels_dir)
    }

    # 2. Clean and recreate destination directories to prevent data leakage from old runs
    for img_dir, lbl_dir in dirs.values():
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        if os.path.exists(lbl_dir):
            shutil.rmtree(lbl_dir)
        create_dirs([img_dir, lbl_dir])

    # 3. Gather files (supporting standard image extensions)
    image_files: List[str] = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(glob.glob(os.path.join(raw_images_dir, ext)))
    
    # Pair images with their corresponding YOLO .txt label files
    paired_data: List[Tuple[str, str]] = []
    for img_path in image_files:
        base_name: str = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path: str = os.path.join(raw_labels_dir, f"{base_name}.txt")
        
        # Only include images that have a corresponding label file
        if os.path.exists(lbl_path):
            paired_data.append((img_path, lbl_path))

    # 4. Shuffle the data securely using the predefined seed
    random.seed(cfg.train.seed)
    random.shuffle(paired_data)

    # 5. Calculate split indices
    total: int = len(paired_data)
    train_ratio: float = cfg.preprocess.split.train
    val_ratio: float = cfg.preprocess.split.val
    
    train_idx: int = int(total * train_ratio)
    val_idx: int = train_idx + int(total * val_ratio)

    splits: Dict[str, List[Tuple[str, str]]] = {
        'train': paired_data[:train_idx],
        'val': paired_data[train_idx:val_idx],
        'test': paired_data[val_idx:]
    }

    # 6. Copy files to their new split directories
    print(f"Splitting dataset: {total} total valid samples.")
    for split_name, files in splits.items():
        if not files:
            continue
            
        img_dest, lbl_dest = dirs[split_name]
        for img_src, lbl_src in files:
            shutil.copy(img_src, os.path.join(img_dest, os.path.basename(img_src)))
            shutil.copy(lbl_src, os.path.join(lbl_dest, os.path.basename(lbl_src)))
            
        print(f"    - {split_name.capitalize()}: {len(files)} samples")

    # 7. Generate and return the YOLO yaml configuration
    return build_yolo_yaml(cfg)