import cv2
import numpy as np
from .morph_cleaned_mask import MorphCleanedMask

class MissingWhite:
    """
    Class for refining binary masks to detect missing white regions in images.
    """
    def process_missing_white(image: np.ndarray, cropout_mask: np.ndarray, white_missing_lower: np.ndarray, white_missing_upper: np.ndarray, morph_opening_size: int, morph_closing_size: int, morph_erosion_size: int) -> np.ndarray:
        """
        Generate a binary mask for detecting white-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with white regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        refined_mask_for_white = cv2.inRange(hsv_image, white_missing_lower, white_missing_upper)
        morphcleaned_refined_white_mask = MorphCleanedMask.morph_cleaned_mask(refined_mask_for_white, morph_opening_size, morph_closing_size, morph_erosion_size) # Apply morphological operations to clean the mask
        combined_refined_mask = cv2.bitwise_or(cropout_mask, morphcleaned_refined_white_mask) # Combine the original mask with the refined mask
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask
        # Prepare HSV parameters for white missing regions
        hsv_morph_parameters_white = {
            'hsv_lower': white_missing_lower,
            'hsv_upper': white_missing_upper,
            'morph_opening_size': morph_opening_size,
            'morph_closing_size': morph_closing_size,
            'morph_erosion_size': morph_erosion_size
        }
        return combined_refined_mask_binary, hsv_morph_parameters_white
