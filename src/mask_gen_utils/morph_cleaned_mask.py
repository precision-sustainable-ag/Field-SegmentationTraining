import numpy as np
import skimage.morphology as morph

class MorphCleanedMask:
    """Class for refining binary masks using morphological operations.
    This class applies opening, closing, and erosion to clean up masks.
    """
    
    def morph_cleaned_mask(mask: np.ndarray, morph_opening_size: int, morph_closing_size: int, morph_erosion_size: int) -> np.ndarray:
        """
        Apply morphological operations (opening, closing, erosion) to clean up the mask.

        Args:
            mask (np.ndarray): Binary mask to refine.

        Returns:
            np.ndarray: Refined binary mask.
        """
        combined_mask = morph.opening(mask, morph.disk(morph_opening_size))
        combined_mask = morph.closing(combined_mask, morph.disk(morph_closing_size))
        combined_mask = morph.erosion(combined_mask, morph.disk(morph_erosion_size))
        return combined_mask
