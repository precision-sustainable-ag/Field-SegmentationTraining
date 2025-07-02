import cv2
import numpy as np

class MissingWhite:
    """
    Class for generating binary masks for detecting white-colored regions in images.
    Uses HSV color space for thresholding.
    """
    
    def process_missing_white(image: np.ndarray, white_missing_lower: np.ndarray, white_missing_upper: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting white-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with white regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_image, white_missing_lower, white_missing_upper)
    