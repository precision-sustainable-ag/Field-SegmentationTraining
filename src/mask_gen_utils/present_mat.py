import cv2
import numpy as np
from omegaconf import DictConfig
from .morph_cleaned_mask import MorphCleanedMask

class PresentMat:
    """
    Class for refining binary masks to detect present mat regions in images.
    """
    def __init__(self, present_mat_cfg: DictConfig):
        self.mat_present_hsv_lower = np.array(present_mat_cfg.hsv_lower, dtype=np.uint8)
        self.mat_present_hsv_upper = np.array(present_mat_cfg.hsv_upper, dtype=np.uint8)
        self.mat_opening_size = present_mat_cfg.opening_kernel_size
        self.mat_closing_size = present_mat_cfg.closing_kernel_size
        self.mat_erosion_size = present_mat_cfg.erosion_kernel_size

    def process(self, image: np.ndarray, cropout_mask: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting mat-present regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with mat-present regions as 0.

        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        refined_mask_for_mat = cv2.bitwise_not(cv2.inRange(hsv_image, self.mat_present_hsv_lower, self.mat_present_hsv_upper))  # Exclude pixels in this range to avoid mat present regions
        
        # Apply morphological operations to clean the mask
        morph_cleaned_mask_for_mat = MorphCleanedMask.morph_cleaned_mask(
            refined_mask_for_mat, 
            self.mat_opening_size, 
            self.mat_closing_size, 
            self.mat_erosion_size
            )  
        
        combined_refined_mask = cv2.bitwise_and(cropout_mask, morph_cleaned_mask_for_mat) # Combine the original mask with the refined mask
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask

        return combined_refined_mask_binary
