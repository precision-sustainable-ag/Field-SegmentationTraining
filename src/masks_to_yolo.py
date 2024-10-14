import logging

from omegaconf import DictConfig

from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# Configure logging
log = logging.getLogger(__name__)



# Example usage
def main(cfg: DictConfig) -> None:

    mask_dir = cfg.paths.masks_dir
    output_dir = cfg.paths.yolo_format_label
        
    convert_segment_masks_to_yolo_seg(masks_dir=mask_dir, output_dir=output_dir, classes=2)
