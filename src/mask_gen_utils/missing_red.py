import cv2
import numpy as np
from omegaconf import DictConfig
from .morph_cleaned_mask import MorphCleanedMask

class MissingRed:
    """
    Class for refining binary masks to detect missing red regions in images.
    """
    
    def __init__(self, missing_red_cfg: DictConfig):
        self.red_missing_hsv_lower = np.array(missing_red_cfg.hsv_lower, dtype=np.uint8)
        self.red_missing_hsv_upper = np.array(missing_red_cfg.hsv_upper, dtype=np.uint8)
        self.red_missing_hsv_lower_2 = np.array(missing_red_cfg.hsv_lower_2, dtype=np.uint8)
        self.red_missing_hsv_upper_2 = np.array(missing_red_cfg.hsv_upper_2, dtype=np.uint8)
        self.red_opening_size = missing_red_cfg.opening_kernel_size
        self.red_closing_size = missing_red_cfg.closing_kernel_size
        self.red_erosion_size = missing_red_cfg.erosion_kernel_size

    def process_missing_red(self, image: np.ndarray, cropout_mask: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting red-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with red-missing regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        refined_mask_for_red_1 = cv2.inRange(hsv_image, self.red_missing_hsv_lower, self.red_missing_hsv_upper)
        refined_mask_for_red_2 = cv2.inRange(hsv_image, self.red_missing_hsv_lower_2, self.red_missing_hsv_upper_2)
        refined_mask_red_final = cv2.bitwise_or(refined_mask_for_red_1, refined_mask_for_red_2)
        
        morphcleaned_refined_mask_red_final = MorphCleanedMask.morph_cleaned_mask(
            refined_mask_red_final, 
            self.red_opening_size, 
            self.red_closing_size, 
            self.red_erosion_size
            )
        
        combined_refined_mask = cv2.bitwise_or(cropout_mask, morphcleaned_refined_mask_red_final)
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8)

        return combined_refined_mask_binary
