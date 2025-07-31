import cv2
import numpy as np
from omegaconf import DictConfig
from .morph_cleaned_mask import MorphCleanedMask

class MissingWhite:
    def __init__(self, missing_white_cfg: DictConfig):
        self.luminance_thresh = missing_white_cfg.luminance_thresh
        self.white_opening_size = missing_white_cfg.opening_kernel_size
        self.white_closing_size = missing_white_cfg.closing_kernel_size
        self.white_erosion_size = missing_white_cfg.erosion_kernel_size
    """
    Class for refining binary masks to detect missing white regions in images.
    """
    def process(self, image: np.ndarray, cropout_mask: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting white-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with white regions as 255.
        """
        # Separate the channels from RGB image
        R = image[:, :, 0].astype(np.float32)
        G = image[:, :, 1].astype(np.float32)
        B = image[:, :, 2].astype(np.float32)

        # Calculate luminance index
        luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B

        # Apply luminance threshold
        if self.luminance_thresh > 0:
            luminance_mask = np.where(luminance > self.luminance_thresh, 255, 0).astype(np.uint8)

        # Morphological cleaning
        morphcleaned_refined_white_mask = MorphCleanedMask.morph_cleaned_mask(
            luminance_mask, 
            self.white_opening_size, 
            self.white_closing_size, 
            self.white_erosion_size
            ) 
        
        # Combine with cropout_mask
        combined_refined_mask = cv2.bitwise_or(cropout_mask, morphcleaned_refined_white_mask) # Combine the original mask with the refined mask

        return combined_refined_mask
