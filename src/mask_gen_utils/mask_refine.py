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
from omegaconf import DictConfig, ListConfig
import pandas as pd
from typing import Dict, Tuple, Any

from src.mask_gen_utils.missing_red  import MissingRed
from src.mask_gen_utils.missing_white import MissingWhite
from src.mask_gen_utils.present_mat import PresentMat
import re

# Logging configuration
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
        
        # Path to the voxel inspection results CSV
        self.voxel_inspection_results_db_path = Path(cfg.paths.voxel_inspection_results_db)
        self.df_voxel_db = self._load_voxel_db() # Load the voxel inspection results into a DataFrame

        # Canonical tag mapping from the configuration
        self.canonical_tag_mapping = cfg.mask_gen.canonical_tag_mapping

        # Tags to remove from the final mask
        self.remove_tags = cfg.mask_gen.remove_tags if cfg.mask_gen.remove_tags else []

        # Only include tags specified in the configuration if provided
        self.only_include_tags = cfg.mask_gen.refine.only_include_tags

        # refine parameters for missing red; HUE for red is split into two ranges to cover the full spectrum
        self.missing_red_cfg = cfg.mask_gen.refine.missing_red
        self.missing_red_processor = MissingRed(self.missing_red_cfg)

        # refine parameters for missing white
        self.missing_white_cfg = cfg.mask_gen.refine.missing_white
        self.missing_white_processor = MissingWhite(self.missing_white_cfg)

        # refine parameters for present mat
        self.present_mat_cfg = cfg.mask_gen.refine.present_mat
        self.present_mat_processor = PresentMat(cfg.mask_gen.refine.present_mat)
    
    def _load_voxel_db(self) -> pd.DataFrame:
        """
        Loads the voxel inspection results CSV into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing voxel inspection results.
        """
        if not self.voxel_inspection_results_db_path.exists():
            raise FileNotFoundError(f"Voxel inspection results database (csv) not found at {self.voxel_inspection_results_db_path}")
        return pd.read_csv(self.voxel_inspection_results_db_path)

    def _get_df_needing_refinement(self, cfg: DictConfig) -> pd.DataFrame:
        """
        Returns a DataFrame of images that need mask refinement.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only images that need refinement.
        """
        # Load tags from config to filter the DataFrame
        tags_to_filter_db = list(cfg.mask_gen.refine.filter_db_by_final_voxel_tags)

        for tag in tags_to_filter_db:
            # Filter out images that have the tag to be filtered in their final_voxel_tag column
            df_voxel_db_need_refining = self.df_voxel_db[~self.df_voxel_db["final_voxel_tag"].astype(str).str.contains(tag, na=False)]
        return df_voxel_db_need_refining

    def _map_voxel_tags(self, df_voxel_db_need_refining: pd.DataFrame) -> Dict[str, str]:
        """
        Maps voxel inspection tags to canonical tags defined in the configuration.
        """
        tag_keywords = self.canonical_tag_mapping

        # Set the voxel tag to the correct canonical tag (the key in tag_keywords)
        tag_map = {}
        for tag_label, keyword in tag_keywords.items():
            # Find all images that match the keyword in their initial_voxel_tag
            matched = df_voxel_db_need_refining[df_voxel_db_need_refining["initial_voxel_tag"].astype(str).str.contains(keyword, na=False)]["image_name"]
            tag_map.update({name: tag_label for name in matched})

        # Check for unmatched tags
        pattern = "|".join([re.escape(keyword) for keyword in tag_keywords.values()])
        unmatched = df_voxel_db_need_refining[~df_voxel_db_need_refining["initial_voxel_tag"].astype(str).str.contains(pattern, na=False, regex=True)]
        unmatched_tag_map = dict(zip(unmatched["image_name"], unmatched["initial_voxel_tag"]))
        if unmatched_tag_map:
            log.warning(f"Unmatched tags found in voxel inspection results: {unmatched_tag_map}")

        if not tag_map:
            raise ValueError("No tags found in voxel inspection results. Ensure the CSV is populated correctly.")
        
        return tag_map
    
    def _update_voxel_tags_to_canonical(self, tag_map: Dict[str, str]) -> None:
        """
        Updates the DataFrame to use canonical tags for each image in tag_map.
        """
        for image_name, canonical_tag in tag_map.items():
            idx = self.df_voxel_db[self.df_voxel_db["image_name"] == image_name].index
            if not idx.empty:
                self.df_voxel_db.loc[idx, "initial_voxel_tag"] = canonical_tag

    def _remove_refined_masks(self, voxel_tag_map: Dict[str, str]) -> Dict[str, str]:
        """
        Removes refined masks for images with tags that are in the remove_tags list.
        """
        copy_of_voxel_tag_map = voxel_tag_map.copy()
        for image_name, canonical_tag in copy_of_voxel_tag_map.items():

            if canonical_tag in self.remove_tags:
                refined_mask_path = self.mask_refine_save_dir / f"{image_name}_mask.png"
                if refined_mask_path.exists():
                    # Remove the refined mask if it exists
                    os.remove(refined_mask_path)
                    log.info(f"Removed mask for tag '{canonical_tag}': {refined_mask_path}")

                else:
                    log.warning(f"Refined mask not found for {image_name} with tag '{canonical_tag}'. No action taken.")

        return voxel_tag_map

    def _process_single_image(self, image_name: str, tag: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single image-mask pair based on the associated tag.

        Args:
            image_name (str): str of the image file name (e.g., "image.jpg").
            tag (str): Issue type tag ("missing_red", "missing_white", "present_mat", "bad", or "other").
        """
        log.info(f"Refining mask for: {image_name}")
        
        # Get the cropout image and initial mask paths
        cropout_image_path = self.cutout_dir / Path(image_name)
        initial_mask_path = self.cutout_dir / str(image_name).replace(".jpg", "_mask.png")

        # Load the cropout image and initial mask
        cropout_image = cv2.cvtColor(cv2.imread(str(cropout_image_path)), cv2.COLOR_BGR2RGB)
        cropout_initial_mask = cv2.imread(str(initial_mask_path), cv2.IMREAD_GRAYSCALE)
    
        if tag == "missing_red":
            log.info("Processing missing red regions.")
            hsv_morph_parameters = self.missing_red_cfg
            refined_mask = self.missing_red_processor.process_missing_red(
                cropout_image, 
                cropout_initial_mask
            )
            
        elif tag == "missing_white":
            log.info("Processing missing white regions.")
            hsv_morph_parameters = self.missing_white_cfg
            refined_mask = self.missing_white_processor.process_missing_white(
                cropout_image, 
                cropout_initial_mask
            )
        
        elif tag == "present_mat":
            log.info("Processing present mat regions.")
            hsv_morph_parameters = self.present_mat_cfg
            refined_mask = self.present_mat_processor.process_present_mat(
                cropout_image, 
                cropout_initial_mask
            )
            
        # Sanity check
        else:
            log.error(f"Unknown tag '{tag}' for image {cropout_image_path}. Skipping refinement.")
            return None, None
        
        return refined_mask, hsv_morph_parameters
    
    def _update_db_with_hsv_parameters(self, image_name: str, hsv_morph_param: Dict[str, Any]) -> None:
        """
        Updates the voxel inspection CSV with the HSV parameters used for mask refinement.
        """

        # Find the row corresponding to the image
        row_idx = self.df_voxel_db[self.df_voxel_db["image_name"] == image_name].index
    
        for key, value in hsv_morph_param.items():            
            if isinstance(value, (list, np.ndarray, ListConfig)):
                value = ','.join(map(str, list(value)))
            self.df_voxel_db.loc[row_idx[0], key] = value

        log.info(f"Updated voxel inspection row for {image_name} with HSV parameters: {hsv_morph_param}")

    
    def _update_db_with_remove_tags(self, tags: Dict[str,str]) -> None:
        """
        Removes entries from the voxel inspection results DataFrame based on specified tags.
        Args:
            tags (Dict[str, str]): Dictionary mapping image names to their canonical tags.
        """
        for image_name, canonical_tag in tags.items():
            if canonical_tag in self.remove_tags:
                log.info(f"Removing {image_name} from voxel inspection results due to tag '{canonical_tag}'.")
                row_idx = self.df_voxel_db[self.df_voxel_db["image_name"] == image_name].index
                self.df_voxel_db.drop(row_idx, inplace=True)
    
    def _save_refined_mask(self, refined_mask: np.ndarray, image_name: str) -> None:
        """
        Saves the refined mask to the specified output path.

        Args:
            refined_mask (np.ndarray): The refined binary mask to save.
            output_path (Path): The path where the mask will be saved.
        """
        mask_output_path = Path(self.mask_refine_save_dir) / str(image_name).replace(".jpg", "_mask.png")
        
        cv2.imwrite(str(mask_output_path), refined_mask)
        log.info(f"Refined mask saved to: {mask_output_path}")

    def _save_df_voxel_db(self) -> None:
        """
        Saves the updated voxel inspection DataFrame to the CSV file.
        """
        self.df_voxel_db.to_csv(self.voxel_inspection_results_db_path, index=False)
        log.info(f"Voxel inspection results database updated and saved to {self.voxel_inspection_results_db_path}")

    def process_cutout_dir(self, cfg: DictConfig) -> None:
        """
        Processes cropout images and refines masks based on voxel inspection tags.
        """
        try:
            ## Handling tags and already refined masks
            # TODO: improve this by catching and handling tag discrepancies, "other" tags, "bad" tags, etc.
            # Get the DataFrame of images that need refinement
            df_voxel_db_need_refining = self._get_df_needing_refinement(cfg)

            # Load and map voxel inspection tags to canonical tags
            tag_map = self._map_voxel_tags(df_voxel_db_need_refining)

            # Update the voxel inspection results DataFrame with canonical tags
            self._update_voxel_tags_to_canonical(tag_map)

            # Remove refined masks for tags that should not be processed
            cleaned_tag_map = self._remove_refined_masks(tag_map)

            # Remove entries from the voxel inspection results DB for tags that should not be processed
            self._update_db_with_remove_tags(cleaned_tag_map)

            for image_name, tag in cleaned_tag_map.items():
                if self.only_include_tags and tag not in self.only_include_tags:
                    log.info(f"Skipping {image_name} with tag '{tag}' as it is not in the only_include_tags list.")
                    continue
                try:
                    # Process each image based on its tag
                    refined_mask, hsv_morph_param = self._process_single_image(image_name, tag)
                    if refined_mask is not None and hsv_morph_param is not None:
                        # Save the refined mask and update the voxel inspection results DB
                        self._update_db_with_hsv_parameters(image_name, hsv_morph_param)
                        self._save_refined_mask(refined_mask, image_name)
                        log.info(f"Refining completed for: {image_name}")
                    else:
                        log.warning(f"Refinement skipped for {image_name} due to processing error or unsupported tag.")
                
                except Exception as e:
                    log.error(f"Error processing {image_name}: {e}", exc_info=True)
        
        except Exception as main_e:
            log.error(f"Fatal error during batch mask refinement: {main_e}", exc_info=True)
        
        finally:
            # Always try to save the DB, even if something failed
            self._save_df_voxel_db()

def main(cfg: DictConfig) -> None:
    """
    Entry point for running the mask refinement process using configured HSV and morphological parameters.

    Args:
        cfg (DictConfig): Hydra configuration object with required settings.

    Processes cropout images and refines masks based on voxel inspection tags.
    TODO: improve this docstring
    TODO: add error handling for missing data, processing errors, etc.
    TODO: do something with masks with "bad", "other" tags (process again, create generic mask, etc.)
    TODO: make tags handling more robust, e.g., handle "other" tags, "bad" tags, tag discrepencies, etc.
    """
    refine_mask = RefineMask(cfg)
    refine_mask.process_cutout_dir(cfg)
    log.info("Refining mask process completed successfully.")
