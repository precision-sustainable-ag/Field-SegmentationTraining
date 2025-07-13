import cv2
import numpy as np
from .morph_cleaned_mask import MorphCleanedMask

class MissingRed:
    """
    Class for refining binary masks to detect missing red regions in images.
    """
    def process_missing_red(image: np.ndarray, cropout_mask: np.ndarray, red_missing_lower_1: np.ndarray, red_missing_upper_1: np.ndarray, red_missing_lower_2: np.ndarray, red_missing_upper_2: np.ndarray, morph_opening_size: int, morph_closing_size: int, morph_erosion_size: int) -> np.ndarray:
        """
        Generate a binary mask for detecting red-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with red-missing regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        refined_mask_for_red_1 = cv2.inRange(hsv_image, red_missing_lower_1, red_missing_upper_1)
        refined_mask_for_red_2 = cv2.inRange(hsv_image, red_missing_lower_2, red_missing_upper_2)
        combined_refined_mask = cv2.bitwise_or(cropout_mask, refined_mask_for_red_1)
        combined_refined_mask = cv2.bitwise_or(combined_refined_mask, refined_mask_for_red_2)
        morphcleaned_combined_refined_mask = MorphCleanedMask.morph_cleaned_mask(combined_refined_mask, morph_opening_size, morph_closing_size, morph_erosion_size)
        morphcleaned_combined_refined_mask_binary = np.where(morphcleaned_combined_refined_mask > 0, 255, 0).astype(np.uint8)

        return morphcleaned_combined_refined_mask_binary
