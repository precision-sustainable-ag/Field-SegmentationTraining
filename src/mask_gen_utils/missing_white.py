import cv2
import numpy as np
from omegaconf import DictConfig
from .morph_cleaned_mask import MorphCleanedMask

class MissingWhite:
    def __init__(self, missing_white_cfg: DictConfig):
        self.white_missing_hsv_lower = np.array(missing_white_cfg.hsv_lower, dtype=np.uint8)
        self.white_missing_hsv_upper = np.array(missing_white_cfg.hsv_upper, dtype=np.uint8)
        self.white_opening_size = missing_white_cfg.opening_kernel_size
        self.white_closing_size = missing_white_cfg.closing_kernel_size
        self.white_erosion_size = missing_white_cfg.erosion_kernel_size
    """
    Class for refining binary masks to detect missing white regions in images.
    """
    def process_missing_white(self, image: np.ndarray, cropout_mask: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting white-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with white regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        refined_mask_for_white = cv2.inRange(hsv_image, self.white_missing_hsv_lower, self.white_missing_hsv_upper)
        
        # Apply morphological operations to clean the mask
        morphcleaned_refined_white_mask = MorphCleanedMask.morph_cleaned_mask(
            refined_mask_for_white, 
            self.white_opening_size, 
            self.white_closing_size, 
            self.white_erosion_size
            ) 
        
        combined_refined_mask = cv2.bitwise_or(cropout_mask, morphcleaned_refined_white_mask) # Combine the original mask with the refined mask
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask
        
        return combined_refined_mask_binary
