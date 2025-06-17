import cv2
import numpy as np

class MissingRed:
    def process_missing_white(self, image: np.ndarray) -> np.ndarray:
            """
            Generate a binary mask for detecting white-colored regions using HSV thresholding.

            Args:
                image (np.ndarray): RGB image.

            Returns:
                np.ndarray: Binary mask with white regions as 255.
            """
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv_image, self.white_missing_lower, self.white_missing_upper)