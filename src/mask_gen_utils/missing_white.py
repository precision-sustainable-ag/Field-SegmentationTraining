import cv2
import numpy as np
from omegaconf import DictConfig
from .morph_cleaned_mask import MorphCleanedMask

class MissingWhite:
    """
    Class for refining binary masks to detect missing white regions in images.
    """

    def __init__(self, missing_white_cfg: DictConfig):
        self.luminance_thresh = missing_white_cfg.luminance_thresh
        self.prune = missing_white_cfg.prune.enabled
    
    def prune_isolated_speckles_fast(self, mask, area_large=10000, radius=300, min_area_keep=30):
        # 0/255 -> 0/1
        m = (mask > 0).astype(np.uint8)

        # Connected components (O(N))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

        # Large comps mask
        large = np.zeros_like(m, dtype=np.uint8)
        large_ids = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= area_large]
        for i in large_ids:
            large[labels == i] = 1

        if large.sum() == 0:
            # No big blobs: just area-filter tiny specks
            keep = np.zeros_like(m)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= min_area_keep:
                    keep[labels == i] = 1
            return (keep * 255).astype(np.uint8)

        # distanceTransform measures distance from non-zero to nearest zero.
        # We want distance to LARGE; so set LARGE=0, others=1.
        inv_large = (1 - large).astype(np.uint8)  # large->0, other->1
        dist = cv2.distanceTransform(inv_large, cv2.DIST_L2, 3)
        near_zone = (dist <= float(radius)).astype(np.uint8)

        # Which labels intersect the near zone?
        labels_near = np.unique(labels[near_zone > 0])
        labels_near = set(int(x) for x in labels_near if x != 0)

        # Build kept mask
        keep = np.zeros_like(m)
        # Keep all large comps
        for i in large_ids:
            keep[labels == i] = 1
        # Keep small comps only if they intersect near_zone and pass min area
        for i in range(1, num):
            if i not in large_ids and stats[i, cv2.CC_STAT_AREA] >= min_area_keep and i in labels_near:
                keep[labels == i] = 1

        return (keep * 255).astype(np.uint8)


    def process(self, image: np.ndarray, cropout_mask: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting white-colored regions using luminance.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with white regions as 255.
        """
        # Separate the channels from RGB image
        R = image[:, :, 0].astype(np.float32)
        G = image[:, :, 1].astype(np.float32)
        B = image[:, :, 2].astype(np.float32)

        # Calculate luminance index
        luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B

        # Apply luminance threshold
        if self.luminance_thresh > 0:
            luminance_mask = np.where(luminance > self.luminance_thresh, 255, 0).astype(np.uint8)

        # Morphological cleaning
        if self.prune:
            pruned_luminance_mask = self.prune_isolated_speckles_fast(luminance_mask, area_large=15000, radius=250, min_area_keep=40)
            luminance_mask = pruned_luminance_mask
        
        # Combine with cropout_mask
        combined_refined_mask = cv2.bitwise_or(cropout_mask, luminance_mask) # Combine the original mask with the refined mask
        combined_refined_mask_binary = np.where(combined_refined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask
        
        return combined_refined_mask_binary
