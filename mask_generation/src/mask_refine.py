"""
Mask Refinement Module 

This module defines the `RefineMask` class which processes segmented images and refines
their corresponding binary masks using HSV thresholding and morphological operations.

Functionality:
- Reads tagged image metadata from a voxel inspection CSV.
- Identifies image issues based on tags:
    * Red-missing (masks where red plant parts are not detected)
    * White-missing (masks where white plant parts are not detected)
    * Mat-present (masks with black/gray mat background present)
    * Bad (masks marked as bad in voxel inspection and need to be removed)
    * Other tags (masks with other issues)
- Applies region-specific refinements and morphological operations.
- Combines original and refined masks for final output.
- Saves the refined masks in a dedicated output directory for downstream use.
"""
import os
import cv2
import logging
import numpy as np
from pathlib import Path
import skimage.morphology as morph
from omegaconf import DictConfig
import pandas as pd

from src.mask_gen_utils.missing_red  import MissingRed
from src.mask_gen_utils.missing_white import MissingWhite
from src.mask_gen_utils.present_mat import PresentMat
from src.mask_gen_utils.morph_cleaned_mask import MorphCleanedMask

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

class RefineMask:
    """
    Refines segmentation masks based on voxel-inspection tags using HSV filtering and 
    morphological operations.

    Supported tags and operations:
    - "missing_red": Detects red regions using HSV and adds to mask.
    - "missing_white": Detects white regions using HSV and adds to mask.
    - "present_mat": Detects background mat presence and removes it from mask.
    - "bad": if present, removes the refined masks so initial mask is used in next steps.
    - "other": skips processing for images with other tags.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the RefineMask processor with configuration parameters.

        Args:
            cfg (DictConfig): Hydra configuration object containing paths and HSV/morph settings.
        """
        # Setup directories
        self.mask_generation_dir = Path(cfg.paths.mask_gen_dir)
        self.developed_images_dir = self.mask_generation_dir / "developed-images"
        self.cutout_dir = self.mask_generation_dir / "cutouts"
        self.mask_refine_save_dir = self.mask_generation_dir / "refined_masks"
        self.mask_refine_save_dir.mkdir(parents=True, exist_ok=True)
        self.voxel_inspection_results_db = Path(cfg.paths.voxel_inspection_results_db)

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
        Processes a single image-mask pair based on the associated tag.
        Combines the original mask with HSV-refined mask and applies cleanup.

        Args:
            cropout_image_path (Path): Path to the cropped RGB image.
            mask_image_path (Path): Path to the initial binary mask.
            tag (str): Issue type tag ("missing_red", "missing_white", "present_mat", "bad", or "other").
        """
        logging.info(f"Refining mask for: {cropout_image_path}")
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
        logging.info(f"Refining completed for: {cropout_image_path}")

    def _load_voxel_tag_map(self) -> dict:
        """
        Loads voxel inspection CSV and returns a mapping of image names to their issue tags.
        """
        if not Path(self.voxel_inspection_results_db).exists():
            logging.exception(f"Voxel inspection CSV not found at {self.voxel_inspection_results_db}")
            return {}

        df = pd.read_csv(self.voxel_inspection_results_db)
        if df.empty:
            logging.exception("Voxel inspection results file is empty.")
            return {}

        tag_keywords = {
            "missing_red": "red",
            "missing_white": "white",
            "present_mat": "mat",
            "bad": "bad",
            "other": "other"
        }

        tag_map = {}
        for tag_label, keyword in tag_keywords.items():
            matched = df[df["voxel tags"].str.contains(keyword, na=False)]["image_name"]
            tag_map.update({name: tag_label for name in matched})

        return tag_map

    def _handle_tagged_image(self, image_path: Path, tag: str) -> None:
        """
        Processes a single image based on its associated tag by either removing its mask or refining it.

        Args:
            image_path (Path): The file path to the image being processed.
            tag (str): The tag associated with the image, indicating how it should be handled.
        Behavior:
            - If the tag is "bad" or "other", removes the corresponding refined mask file if it exists and logs the action.
            - If the tag is not one of {"missing_red", "missing_white", "present_mat"}, raises a ValueError to halt processing.
            - For valid tags, selects the refined mask if available; otherwise, uses the default mask, and processes the image accordingly.
        Raises:
            ValueError: If the tag is not recognized as a valid processing tag.
        """
        stem = image_path.stem
        mask_filename = f"{stem}_mask.png"
        refined_mask_path = self.mask_refine_save_dir / mask_filename
        default_mask_path = self.cutout_dir / mask_filename

        # Remove mask for "bad" or "other" tags and return early
        if tag in {"bad", "other"}:
            if refined_mask_path.exists():
                os.remove(refined_mask_path)
                logging.info(f"Removed mask for tag '{tag}': {refined_mask_path}")
            return

        # Validate tag
        if tag not in {"missing_red", "missing_white", "present_mat"}:
            raise ValueError(f"Unknown tag '{tag}' encountered for image {image_path}. Stopping processing.")

        # Prefer refined mask if it exists, else use default
        mask_path = refined_mask_path if refined_mask_path.exists() else default_mask_path
        self.process_single_image(image_path, mask_path, tag)

    def process_cutout_dir(self) -> None:
        """
        Iterates over all cropout images and applies refinement if they are tagged.
        Images without voxel tags are skipped.
        """
        logging.info(f"Processing images in folder: {self.cutout_dir} with issues from voxel inspection.")
        tag_map = self._load_voxel_tag_map()
        if not tag_map:
            return

        for image_path in sorted(self.cutout_dir.glob("*.jpg")):
            tag = tag_map.get(image_path.name)
            if not tag:
                continue
            self._handle_tagged_image(image_path, tag)

def main(cfg: DictConfig) -> None:
    """
    Entry point for running the mask refinement process using configured HSV and morphological parameters.

    Args:
        cfg (DictConfig): Hydra configuration object with required settings.
    """
    refine_mask = RefineMask(cfg)
    refine_mask.process_cutout_dir()
    logging.info("Refining mask process completed successfully.")
