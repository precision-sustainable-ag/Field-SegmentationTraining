import cv2
import numpy as np
from .morph_cleaned_mask import MorphCleanedMask

class PresentMat:
    """
    Class for refining binary masks to detect present mat regions in images.
    """
    def process_present_mat(image: np.ndarray, cropout_mask: np.ndarray, mat_present_lower: np.ndarray, mat_present_upper: np.ndarray, morph_opening_size: int, morph_closing_size: int, morph_erosion_size: int) -> np.ndarray:
        """
        Generate a binary mask for detecting mat-present regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with mat-present regions as 0.

        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        refined_mask_for_mat = cv2.bitwise_not(cv2.inRange(hsv_image, mat_present_lower, mat_present_upper))  # Exclude pixels in this range to avoid mat present regions
        morph_cleaned_mask_for_mat = MorphCleanedMask.morph_cleaned_mask(refined_mask_for_mat, morph_opening_size, morph_closing_size, morph_erosion_size)  # Apply morphological operations to clean the mask
        combined_refined_mask = cv2.bitwise_and(cropout_mask, morph_cleaned_mask_for_mat) # Combine the original mask with the refined mask
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask
        # Prepare HSV parameters for mat present regions
        hsv_morph_parameters_mat = {
            'hsv_lower': mat_present_lower,
            'hsv_upper': mat_present_upper,
            'morph_opening_size': morph_opening_size,
            'morph_closing_size': morph_closing_size,
            'morph_erosion_size': morph_erosion_size
        }
        return combined_refined_mask_binary, hsv_morph_parameters_mat
