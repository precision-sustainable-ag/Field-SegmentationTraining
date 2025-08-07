import cv2
import numpy as np
from omegaconf import DictConfig
from .morph_cleaned_mask import MorphCleanedMask

class MissingWhite:
    """
    Class for refining binary masks to detect missing white regions in images.
    """

    def __init__(self, missing_white_cfg: DictConfig):
        self.luminance_thresh = missing_white_cfg.luminance_thresh
        self.white_opening_size = missing_white_cfg.opening_kernel_size
        self.white_closing_size = missing_white_cfg.closing_kernel_size
        self.white_erosion_size = missing_white_cfg.erosion_kernel_size
    
    def process(self, image: np.ndarray, cropout_mask: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting white-colored regions using luminance.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with white regions as 255.
        """
        # Separate the channels from RGB image
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        # Normalize the channels
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # Linearize the RGB values to convert the default gamma-corrected RGB values to linear RGB values.
        r_lin = np.where(r_norm < 0.04045, r_norm / 12.92, ((r_norm + 0.055) / 1.055) ** 2.4)
        g_lin = np.where(g_norm < 0.04045, g_norm / 12.92, ((g_norm + 0.055) / 1.055) ** 2.4)
        b_lin = np.where(b_norm < 0.04045, b_norm / 12.92, ((b_norm + 0.055) / 1.055) ** 2.4)

        # Calculate luminance index
        luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

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
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask
        
        return combined_refined_mask_binary
