import cv2
import numpy as np
from omegaconf import DictConfig
from .morph_cleaned_mask import MorphCleanedMask

class MissingRed:
    """
    Class for refining binary masks to detect missing red regions in images.
    """
    
    def __init__(self, missing_red_cfg: DictConfig):
        self.exr_thresh = missing_red_cfg.exr_thresh  # default if not set
        self.saturation_thresh = missing_red_cfg.saturation_thresh  # default to 0 to always boost, but can be >0    
        self.red_opening_size = missing_red_cfg.opening_kernel_size
        self.red_closing_size = missing_red_cfg.closing_kernel_size
        self.red_erosion_size = missing_red_cfg.erosion_kernel_size

    def process_missing_red(self, image: np.ndarray, cropout_mask: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting red-colored regions using Excess Red (ExR) boosted by saturation.

        Args:
            image (np.ndarray): RGB image (OpenCV format).
            cropout_mask (np.ndarray): Pre-existing mask to combine.

        Returns:
            np.ndarray: Binary mask with red-missing regions as 255.
        """
        # Calculate ExR
        R = image[:, :, 0].astype(np.float32)
        G = image[:, :, 1].astype(np.float32)
        B = image[:, :, 2].astype(np.float32)
        exr = 2 * R - G - B

        # Calculate normalized saturation and mask out very gray pixels if desired
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[...,1].astype(np.float32)
        s_norm = s / 255.0
        if self.saturation_thresh > 0:
            color_mask = (s >= self.saturation_thresh).astype(np.float32)
            s_norm = s_norm * color_mask  # Optional: set normalized S to zero if below threshold

        # Boost ExR by normalized saturation
        exr_boosted = exr * s_norm

        # Threshold ExR (boosted)
        exr_mask = np.where(exr_boosted > self.exr_thresh, 255, 0).astype(np.uint8)

        # Morphological cleaning
        morphcleaned_exr_mask = MorphCleanedMask.morph_cleaned_mask(
            exr_mask,
            self.red_opening_size,
            self.red_closing_size,
            self.red_erosion_size
        )

        # Combine with cropout_mask
        combined_refined_mask = cv2.bitwise_or(cropout_mask, morphcleaned_exr_mask)
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8)

        return combined_refined_mask_binary