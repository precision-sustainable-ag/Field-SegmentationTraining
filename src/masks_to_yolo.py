import cv2
import shutil
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# Configure logging
log = logging.getLogger(__name__)

# Example usage
def main(cfg: DictConfig) -> None:
    """
    Main function to convert segmentation masks to YOLO format.
    Args:
        cfg (DictConfig): Configuration object containing paths for masks directory and YOLO format label output directory.
    Returns:
        None
    """
    log.info("Converting segment masks to YOLO format.")
    # Get paths from configuration
    resized_mask_dir = Path(cfg.paths.cropped_resized_masks_dir)
    output_dir = Path(cfg.paths.yolo_format_label)
    modified_mask_dir = Path(cfg.paths.modified_masks_dir)
    
    modified_mask_dir.mkdir(parents=True, exist_ok=True)

    log.info("Assigning 0 to background and 1 to mask.")
    for mask_path in tqdm(resized_mask_dir.iterdir(), desc="Processing masks", unit="mask"):
        mask = cv2.imread(str(mask_path))
        mask_modified = np.where(mask == 255, 0, 1).astype(np.uint8) # assign 0 to background and 1 mask
        cv2.imwrite(str(modified_mask_dir / mask_path.name), mask_modified)
        
    # Convert segment masks to YOLO format
    convert_segment_masks_to_yolo_seg(masks_dir=modified_mask_dir, output_dir=output_dir, classes=1)
    # remove the modified mask directory
    shutil.rmtree(modified_mask_dir)
    log.info("Finished converting segment masks to YOLO format.")