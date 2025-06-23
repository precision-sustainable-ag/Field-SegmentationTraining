"""
Mask Refinement Module for Post-Segmentation Cleanup

This module defines the `RefineMask` class which processes segmented images and refines
their corresponding binary masks using HSV thresholding and morphological operations.
It reads tagged image metadata from a CSV (voxel inspection results) and applies the
appropriate refinement based on whether the issue is red-missing, white-missing, or mat-present.

The refined masks are saved in a dedicated output directory for downstream use.
"""
import cv2
import logging
import numpy as np
from pathlib import Path
import skimage.morphology as morph
from omegaconf import DictConfig
import pandas as pd

from mask_gen_utils.missing_red  import MissingRed
from mask_gen_utils.missing_white import MissingWhite
from mask_gen_utils.present_mat import PresentMat
from mask_gen_utils.morph_cleaned_mask import MorphCleanedMask

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

class RefineMask:
    """
    Class for refining segmentation masks based on identified issues using HSV filtering and 
    morphological operations.

    This class supports detecting and correcting common segmentation mask problems such as:
    - Missing red regions (e.g., red targets not detected)
    - Missing white regions (e.g., white targets or tape)
    - Mat presence (e.g., black/gray mat detected in background)

    It performs:
    - HSV thresholding to identify issue regions
    - Morphological cleanup (opening, closing, erosion)
    - Merging refined masks with original segmentation masks
    - Batch processing of images tagged via voxel inspection
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the RefineMask processor with configuration parameters.

        Args:
            cfg (DictConfig): Hydra configuration object containing paths and HSV/morph settings.
        """
        # Setup directories
        self.mask_generation_dir = Path(cfg.paths.mask_generation_dir)
        self.developed_images_dir = self.mask_generation_dir / "developed-images"
        self.cutout_dir = self.mask_generation_dir / "cutouts"
        self.mask_refine_save_dir = self.mask_generation_dir / "refined_masks"
        self.mask_refine_save_dir.mkdir(parents=True, exist_ok=True)
        self.voxel_inspection_results_csv = Path(cfg.paths.voxel_inspection_results_csv)

        # Initialize variables for image and mask processing
        self.cropout_image = None
        self.cropout_mask = None

        # HSV thresholds for different color masks
        self.red_missing_lower = np.array(cfg.mask_refine.hsv_missing_red.lower, dtype=np.uint8)
        self.red_missing_upper = np.array(cfg.mask_refine.hsv_missing_red.upper, dtype=np.uint8)
        self.white_missing_lower = np.array(cfg.mask_refine.hsv_missing_white.lower, dtype=np.uint8)
        self.white_missing_upper = np.array(cfg.mask_refine.hsv_missing_white.upper, dtype=np.uint8)
        self.mat_present_lower = np.array(cfg.mask_refine.hsv_present_mat.lower, dtype=np.uint8)
        self.mat_present_upper = np.array(cfg.mask_refine.hsv_present_mat.upper, dtype=np.uint8)

        # Morphological operation parameters
        self.morph_opening_size = cfg.mask_refine.opening_kernel_size
        self.morph_closing_size = cfg.mask_refine.closing_kernel_size
        self.morph_erosion_size = cfg.mask_refine.erosion_kernel_size
        self.exg_threshold = cfg.mask_refine.exg_threshold
    
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

    def process_present_mat(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting mat-present (gray or black) regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with mat-present regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_image, self.mat_present_lower, self.mat_present_upper)

    def process_missing_red(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for detecting red-colored regions using HSV thresholding.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Binary mask with red-missing regions as 255.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_image, self.red_missing_lower, self.red_missing_upper)

    def morph_cleaned_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations (opening, closing, erosion) to clean up the mask.

        Args:
            mask (np.ndarray): Binary mask to refine.

        Returns:
            np.ndarray: Refined binary mask.
        """
        combined_mask = morph.opening(mask, morph.disk(self.morph_opening_size))
        combined_mask = morph.closing(combined_mask, morph.disk(self.morph_closing_size))
        combined_mask = morph.erosion(combined_mask, morph.disk(self.morph_erosion_size))
        return combined_mask

    def process_single_image(self, cropout_image_path: Path, mask_image_path: Path, tag: str) -> None:
        """
        Processes a single image and its mask by combining it with a red-missing HSV mask,
        applying morphological refinements, and saving the final result.

        Args:
            cropout_image_path (Path): Path to the original cropped RGB image.
            mask_image_path (Path): Path to the initial binary mask.
        """
        logging.info(f"Starting post-segmentation processing for: {cropout_image_path}")
        self.cropout_image = cv2.cvtColor(cv2.imread(str(cropout_image_path)), cv2.COLOR_BGR2RGB)
        self.cropout_mask = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)

        if tag == "missing_red":
            logging.info("Processing missing red regions.")
            refined_mask = MissingRed.process_missing_red(self.cropout_image, self.red_missing_lower, self.red_missing_upper)
        elif tag == "missing_white":
            logging.info("Processing missing white regions.")
            refined_mask = MissingWhite.process_missing_white(self.cropout_image, self.white_missing_lower, self.white_missing_upper)
        elif tag == "present_mat":
            logging.info("Processing present mat regions.")
            refined_mask = PresentMat.process_present_mat(self.cropout_image, self.mat_present_lower, self.mat_present_upper)

        combined_mask = cv2.bitwise_or(self.cropout_mask, refined_mask) # Combine the original mask with the refined mask
        combined_mask = MorphCleanedMask.morph_cleaned_mask(combined_mask, self.morph_opening_size, self.morph_closing_size, self.morph_erosion_size) # Apply morphological operations to clean the mask
        combined_mask = np.where(combined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask

        output_image_path = self.mask_refine_save_dir / cropout_image_path.name
        mask_output_path = Path(str(output_image_path).replace(".jpg", "_mask.png"))

        logging.info(f"Saving final mask to: {mask_output_path}")
        cv2.imwrite(str(mask_output_path), combined_mask)
        logging.info(f"Mask generation complete for: {cropout_image_path}")

    def process_cutout_dir(self) -> None:
        """
        Processes all crop-out images and their corresponding masks in the cutout directory,
        refines each mask, and saves the output to the specified directory.
        """
        logging.info(f"Processing all images in folder: {self.cutout_dir} that have been tagged with issues in the voxel inspection results.")
        # read db
        df_voxel_results = pd.read_csv(self.voxel_inspection_results_csv)
        if df_voxel_results.empty:
            logging.exception(f"No voxel inspection results found in {self.voxel_inspection_results_csv}.")
            return
        
        # Create lists of file names based on tags in the DataFrame
        missing_red_images = df_voxel_results[df_voxel_results['tags'].str.contains("red", na=False)]['image_name'].tolist()
        missing_white_images = df_voxel_results[df_voxel_results['tags'].str.contains("white", na=False)]['image_name'].tolist()
        present_mat_images = df_voxel_results[df_voxel_results['tags'].str.contains("mat", na=False)]['image_name'].tolist()

        # Match and process image-mask pairs based on stem names and tags
        for image_path in sorted(self.cutout_dir.glob("*.jpg")):
            tag = None
            if image_path.name in missing_red_images:
                tag = "missing_red"
            elif image_path.name in missing_white_images:
                tag = "missing_white"
            elif image_path.name in present_mat_images:
                tag = "present_mat"

            if tag is not None:
                stem = image_path.stem
                mask_path = self.cutout_dir / f"{stem}_mask.png"
                self.process_single_image(image_path, mask_path, tag)

def main(cfg: DictConfig) -> None:
    """
    Entry point for running the mask refinement process using configured HSV and morphological parameters.

    Args:
        cfg (DictConfig): Hydra configuration object with required settings.
    """
    refine_mask = RefineMask(cfg)
    refine_mask.process_cutout_dir()
    logging.info("Refining mask process completed successfully.")
