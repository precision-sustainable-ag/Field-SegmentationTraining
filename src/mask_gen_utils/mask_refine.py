"""
Mask Refinement Module

Refines binary segmentation masks for plant images based on tags from a voxel inspection CSV.

Functionality:
- Identifies mask issues using tags: "missing_red", "missing_white", "present_mat", "bad", and "other".
- Applies HSV thresholding and morphological operations to fix missing regions or remove background mats.
- Updates the voxel inspection CSV with applied parameters.
- Saves refined masks to an output directory for downstream use.
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
        self.mask_generation_dir = Path(cfg.paths.project_maskgen_dir)
        self.developed_images_dir = self.mask_generation_dir / "developed-images"
        self.cutout_dir = self.mask_generation_dir / "cutouts"
        self.mask_refine_save_dir = self.mask_generation_dir / "refined_masks"
        self.mask_refine_save_dir.mkdir(parents=True, exist_ok=True)
        self.voxel_inspection_results_db_path = Path(cfg.paths.voxel_inspection_results_db)
        self.df_voxel_db = pd.read_csv(self.voxel_inspection_results_db_path)

        # Initialize variables for image and mask processing
        self.cropout_image_path = None
        self.cropout_image = None
        self.cropout_mask = None

        # refine parameters for missing red; HUE for red is split into two ranges to cover the full spectrum
        self.red_missing_hsv_lower = np.array(cfg.mask_gen.refine.missing_red.hsv_lower_1, dtype=np.uint8)
        self.red_missing_hsv_upper = np.array(cfg.mask_gen.refine.missing_red.hsv_upper_1, dtype=np.uint8)
        self.red_missing_hsv_lower_2 = np.array(cfg.mask_gen.refine.missing_red.hsv_lower_2, dtype=np.uint8)
        self.red_missing_hsv_upper_2 = np.array(cfg.mask_gen.refine.missing_red.hsv_upper_2, dtype=np.uint8)
        self.red_opening_size = cfg.mask_gen.refine.missing_red.opening_kernel_size
        self.red_closing_size = cfg.mask_gen.refine.missing_red.closing_kernel_size
        self.red_erosion_size = cfg.mask_gen.refine.missing_red.erosion_kernel_size

        # refine parameters for missing white
        self.white_missing_hsv_lower = np.array(cfg.mask_gen.refine.missing_white.hsv_lower, dtype=np.uint8)
        self.white_missing_hsv_upper = np.array(cfg.mask_gen.refine.missing_white.hsv_upper, dtype=np.uint8)
        self.white_opening_size = cfg.mask_gen.refine.missing_white.opening_kernel_size
        self.white_closing_size = cfg.mask_gen.refine.missing_white.closing_kernel_size
        self.white_erosion_size = cfg.mask_gen.refine.missing_white.erosion_kernel_size

        # refine parameters for present mat
        self.mat_present_hsv_lower = np.array(cfg.mask_gen.refine.present_mat.hsv_lower, dtype=np.uint8)
        self.mat_present_hsv_upper = np.array(cfg.mask_gen.refine.present_mat.hsv_upper, dtype=np.uint8)
        self.mat_opening_size = cfg.mask_gen.refine.present_mat.opening_kernel_size
        self.mat_closing_size = cfg.mask_gen.refine.present_mat.closing_kernel_size
        self.mat_erosion_size = cfg.mask_gen.refine.present_mat.erosion_kernel_size
    
    def process_single_image(self, mask_image_path: Path, tag: str) -> None:
        """
        Processes a single image-mask pair based on the associated tag.
        Combines the original mask with HSV-refined mask and applies cleanup and update to the voxel inspection database.

        Args:
            image_path (Path): Path to the cropped RGB image.
            mask_image_path (Path): Path to the initial binary mask.
            tag (str): Issue type tag ("missing_red", "missing_white", "present_mat", "bad", or "other").
        """
        logging.info(f"Refining mask for: {self.cropout_image_path}")
        self.cropout_image = cv2.cvtColor(cv2.imread(str(self.cropout_image_path)), cv2.COLOR_BGR2RGB)
        self.cropout_mask = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)
        output_image_path = self.mask_refine_save_dir / self.cropout_image_path.name
        mask_output_path = Path(str(output_image_path).replace(".jpg", "_mask.png"))

        if tag == "missing_red":
            log.info("Processing missing red regions.")
            refined_mask, self.hsv_morph_parameters = MissingRed.process_missing_red(
                self.cropout_image, 
                self.cropout_mask, 
                self.red_missing_hsv_lower, 
                self.red_missing_hsv_upper, 
                self.red_missing_hsv_lower_2, 
                self.red_missing_hsv_upper_2, 
                self.red_opening_size, 
                self.red_closing_size, 
                self.red_erosion_size
            )
            self.update_db_with_hsv_parameters()
            # Remove previous mask if it exists
            if mask_output_path.exists():
                os.remove(mask_output_path) # Delete previous mask if it exists
        elif tag == "missing_white":
            log.info("Processing missing white regions.")
            refined_mask, self.hsv_morph_parameters = MissingWhite.process_missing_white(
                self.cropout_image, 
                self.cropout_mask, 
                self.white_missing_hsv_lower, 
                self.white_missing_hsv_upper, 
                self.white_opening_size, 
                self.white_closing_size, 
                self.white_erosion_size
            )
            self.update_db_with_hsv_parameters()
            # Remove previous mask if it exists
            if mask_output_path.exists():
                os.remove(mask_output_path) # Delete previous mask if it exists        
        elif tag == "present_mat":
            log.info("Processing present mat regions.")
            refined_mask, self.hsv_morph_parameters = PresentMat.process_present_mat(
                self.cropout_image, 
                self.cropout_mask, 
                self.mat_present_hsv_lower, 
                self.mat_present_hsv_upper, 
                self.mat_opening_size, 
                self.mat_closing_size, 
                self.mat_erosion_size
            )
            self.update_db_with_hsv_parameters()
            # Remove previous mask if it exists
            if mask_output_path.exists():
                os.remove(mask_output_path) # Delete previous mask if it exists
        else:
            log.warning(f"Unknown tag '{tag}' for image {self.cropout_image_path}. Skipping refinement.")
            return
        
        logging.info(f"Saving final mask to: {mask_output_path}")
        cv2.imwrite(str(mask_output_path), refined_mask)
        logging.info(f"Refining completed for: {self.cropout_image_path}")

    def _load_voxel_tag_map(self) -> dict:
        """
        Loads voxel inspection CSV and returns a mapping of image names to their issue tags.
        """
        tag_keywords = {
            "missing_red": "red",
            "missing_white": "white",
            "present_mat": "mat",
            "bad": "bad",
            "other": "other"
        }

        tag_map = {}
        for tag_label, keyword in tag_keywords.items():
            matched = self.df_voxel_db[self.df_voxel_db["voxel tags"].str.contains(keyword, na=False)]["image_name"]
            tag_map.update({name: tag_label for name in matched})

        return tag_map

    def update_db_with_hsv_parameters(self) -> None:
        """
        Updates the voxel inspection CSV with the HSV and morphological parameters used for mask refinement
        for the current image only.
        """
        # Get the current image name
        image_name = Path(self.cropout_image_path).name if isinstance(self.cropout_image_path, Path) else None
        if image_name is None:
            logging.warning("No image name found to update HSV parameters.")
            return

        # Find the row corresponding to the image
        row_idx = self.df_voxel_db[self.df_voxel_db["image_name"] == image_name].index
        if row_idx.empty:
            logging.warning(f"Image {image_name} not found in voxel inspection DB.")
            return

        for key, value in self.hsv_morph_parameters.items():
            if isinstance(value, (list, np.ndarray)):
                value = ','.join(map(str, value))
            self.df_voxel_db.loc[row_idx, key] = value

        self.df_voxel_db.to_csv(self.voxel_inspection_results_db_path, index=False)
        logging.info(f"Updated voxel inspection row for {image_name} with HSV parameters: {self.hsv_morph_parameters}")

    def _handle_tagged_image(self, tag: str) -> None:
        """
        Processes a single image based on its associated tag by either removing its mask or refining it.

        Args:
            image_path (Path): The file path to the image being processed.
            tag (str): The tag associated with the image, indicating how it should be handled.
        Raises:
            ValueError: If the tag is not recognized as a valid processing tag.
        """
        stem = self.cropout_image_path.stem
        mask_filename = f"{stem}_mask.png"
        refined_mask_path = self.mask_refine_save_dir / mask_filename
        initial_mask_path = self.cutout_dir / mask_filename

        if tag in {"bad", "other"}: # mask with bad tag gets removed ##### figure out OTHER tag
            os.remove(refined_mask_path) if refined_mask_path.exists() else None
            logging.info(f"Removed mask for tag '{tag}': {refined_mask_path}")
        elif tag in {"missing_red", "missing_white", "present_mat"}:
            # Prefer refined mask if it exists, else use default
            self.process_single_image(initial_mask_path, tag)
        else:
            raise ValueError(f"Unknown tag '{tag}' encountered for image {self.cropout_image_path}. Stopping processing.")

    def process_cutout_dir(self) -> None:
        """
        Iterates over all cropout images and applies refinement if they are tagged.
        Images without voxel tags are skipped.
        """
        logging.info(f"Processing images in folder: {self.cutout_dir} with issues from voxel inspection.")
        tag_map = self._load_voxel_tag_map()
        if not tag_map:
            return

        for path in sorted(self.cutout_dir.glob("*.jpg")):
            self.cropout_image_path = path
            tag = tag_map.get(self.cropout_image_path.name)
            if not tag:
                continue
            self._handle_tagged_image(tag)

def main(cfg: DictConfig) -> None:
    """
    Entry point for running the mask refinement process using configured HSV and morphological parameters.

    Args:
        cfg (DictConfig): Hydra configuration object with required settings.
    """
    refine_mask = RefineMask(cfg)
    refine_mask.process_cutout_dir()
    logging.info("Refining mask process completed successfully.")
