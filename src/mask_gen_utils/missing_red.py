import cv2
import numpy as np

class MissingRed:
    """
    Class for generating binary masks for red-colored regions in images.
    Uses HSV color space for thresholding.
    """

    def process_missing_red(image: np.ndarray, red_missing_lower: np.ndarray, red_missing_upper: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting red-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with red-missing regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_image, red_missing_lower, red_missing_upper)
