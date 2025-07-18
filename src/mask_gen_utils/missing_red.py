import cv2
import numpy as np
from .morph_cleaned_mask import MorphCleanedMask

class MissingRed:
    """
    Class for refining binary masks to detect missing red regions in images.
    """
    def process_missing_red(image: np.ndarray, cropout_mask: np.ndarray, red_missing_lower: np.ndarray, red_missing_upper: np.ndarray, red_missing_lower_2: np.ndarray, red_missing_upper_2: np.ndarray, morph_opening_size: int, morph_closing_size: int, morph_erosion_size: int) -> np.ndarray:
        """
        Generate a binary mask for detecting red-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with red-missing regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        refined_mask_for_red_1 = cv2.inRange(hsv_image, red_missing_lower, red_missing_upper)
        refined_mask_for_red_2 = cv2.inRange(hsv_image, red_missing_lower_2, red_missing_upper_2)
        refined_mask_red_final = cv2.bitwise_or(refined_mask_for_red_1, refined_mask_for_red_2)
        morphcleaned_refined_mask_red_final = MorphCleanedMask.morph_cleaned_mask(refined_mask_red_final, morph_opening_size, morph_closing_size, morph_erosion_size)
        combined_refined_mask = cv2.bitwise_or(cropout_mask, morphcleaned_refined_mask_red_final)
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8)
        # Prepare HSV parameters for red missing regions
        hsv_morph_parameters_red = {
            'hsv_lower': red_missing_lower,
            'hsv_upper': red_missing_upper,
            'hsv_lower_2': red_missing_lower_2,
            'hsv_upper_2': red_missing_upper_2,
            'morph_opening_size': morph_opening_size,
            'morph_closing_size': morph_closing_size,
            'morph_erosion_size': morph_erosion_size
        }
        return combined_refined_mask_binary, hsv_morph_parameters_red
