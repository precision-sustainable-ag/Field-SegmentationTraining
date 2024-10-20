import logging
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
    # Get paths from configuration
    mask_dir = cfg.paths.masks_dir
    output_dir = cfg.paths.yolo_format_label
    
    log.info("Converting segment masks to YOLO format.")
    convert_segment_masks_to_yolo_seg(masks_dir=mask_dir, output_dir=output_dir, classes=2)
    log.info("Finished converting segment masks to YOLO format.")