import cv2
import numpy as np

class PresentMat:
    """Class for generating a binary mask for black/gray regions of mat-present in an image.
    Uses HSV color space for thresholding.
    """
    
    def process_present_mat(image: np.ndarray, mat_present_lower: np.ndarray, mat_present_upper: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting mat-present (gray or black) regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with mat-present regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_image, mat_present_lower, mat_present_upper)
