import cv2
import logging
import numpy as np
from pathlib import Path
import skimage.morphology as morph
from utils.utils_post_seg import make_exg
from omegaconf import DictConfig

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

class PostSegmentationProcessing:
    """
    Class for post-segmentation image processing using ExG thresholding and white pixel filtering.

    This pipeline includes:
    - Loading an image
    - Extracting vegetation (ExG) and white pixel masks
    - Combining and refining masks
    - Applying the final mask to the original image
    - Saving the output
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the processor with the path to the image.

        Args:
            image_path (Path): Path to the input image.
        """
        # Set up output directories
        self.mask_generation_dir = Path(cfg.paths.mask_generation_dir)
        self.developed_images_dir = self.mask_generation_dir / "developed-images"
        self.cutout_dir = self.mask_generation_dir / "cutouts"
        self.post_processing_save_dir = self.mask_generation_dir / "post_segment_processing"
        self.post_processing_save_dir.mkdir(parents=True, exist_ok=True)

        # Set up images and masks
        self.cropout_image = None
        self.exg_mask = None
        self.white_mask = None
        self.gray_mask = None
        self.non_green_stem_mask = None
        self.red_rgb_mask = None

    def generate_exg_mask(self, threshold: int = 20):
        """
        Generate a binary mask from the Excess Green (ExG) index.
        """
        exg = make_exg(self.image)
        _, exg_mask = cv2.threshold(exg, threshold, 255, cv2.THRESH_BINARY)
        return exg_mask

    def generate_white_mask(self):
        """
        Generate a mask for detecting white-colored regions in the image.
        """
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        return white_mask

    def generate_gray_mask(self):
        """
        Generate a gray mask from the original image.
        """
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([15, 54, 46])
        upper_gray = np.array([21, 94, 150])
        gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
        return gray_mask

    def generate_non_green_stem_mask_reddish(self, image: np.ndarray):
        """
        Generate a non-green stem mask from the original image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Lower red
        red_lower1 = (0, 100, 30) # works best: (0, 100, 40)
        red_upper1 = (179, 255, 255) # works best: (179, 255, 255)
        non_green_stem_mask = cv2.inRange(hsv_image, red_lower1, red_upper1)

        return non_green_stem_mask

    def refine_mask(self, mask):
        """
        Apply morphological operations to clean up the combined mask.
        """
        combined_mask = morph.opening(mask, morph.disk(3))
        combined_mask = morph.closing(mask, morph.disk(4))
        combined_mask = morph.erosion(mask, morph.disk(3))
        return combined_mask
    
    def plot_side_by_side(self, title1: str, image1: np.ndarray, title2: str, image2: np.ndarray):
        """
        Plot two images side by side for comparison.
        
        Args:
            title1 (str): Title for the first image.
            image1 (np.ndarray): First image to display.
            title2 (str): Title for the second image.
            image2 (np.ndarray): Second image to display.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(title1)
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(title2)
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(str(self.output_image_path).replace(".jpg", "_comparison.png"))

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray):
        """
        Overlay the mask on the original image for visualization.

        Args:
            mask (np.ndarray): The binary mask to overlay.
        
        Returns:
            np.ndarray: The image with the mask overlay.
        """
        # Create a red mask where mask > 0
        red_mask = np.zeros_like(image)
        red_mask[mask > 0] = [0, 0, 255]  # BGR for red
        return cv2.addWeighted(self.image, 0.7, red_mask, 0.7, 0)
    
    def process_single_image(self, cropout_image_path: Path, mask_image_path: Path):
        """
        Main method to run the full post-segmentation processing pipeline.
        """
        logging.info(f"Starting post-segmentation processing for: {self.cropout_image}")
        self.cropout_image = cv2.cvtColor(cv2.imread(str(cropout_image_path)), cv2.COLOR_BGR2RGB)
        self.cropout_mask = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)

        # self.exg_mask = self.generate_exg_mask()
        # self.white_mask = self.generate_white_mask()
        # self.gray_mask = self.generate_gray_mask()
        self.non_green_stem_mask = self.generate_non_green_stem_mask_reddish(self.cropout_image)

        combined_mask = cv2.bitwise_or(self.cropout_mask, self.non_green_stem_mask)
        combined_mask = self.refine_mask(combined_mask)
        combined_mask = np.where(combined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask

        # final_cutout_image = cv2.bitwise_and(self.image, self.image, mask=self.combined_mask)
        final_cutout_image = cv2.bitwise_and(self.cropout_image, self.cropout_image, mask=combined_mask)

        # Image and mask paths for saving
        output_image_path = self.post_processing_save_dir / cropout_image_path.name
        mask_output_path = Path(str(output_image_path).replace(".jpg", "_mask.png"))

        # Plot the original image and final mask side by side
        self.plot_side_by_side("Original Image", self.cropout_image, "Final Mask", final_cutout_image)

        # Overlay the mask on the original image and save it
        overlaid_image = self.overlay_mask(self.cropout_image, combined_mask)
        cv2.imwrite(str(output_image_path).replace(".jpg", "_overlaid.jpg"), overlaid_image)

        # Save the final mask
        logging.info(f"Saving final mask to: {mask_output_path}")
        cv2.imwrite(str(mask_output_path), combined_mask)

        # Save the final cutout image
        # logging.info(f"Saving final image to: {self.output_image_path}")
        # cv2.imwrite(str(self.output_image_path), final_cutout_image)
        logging.info(f"Post-segmentation processing complete for: {self.cropout_image}")

    def process_cutout_dir(self):
        """
        Process all image and mask pairs in the input folder using PostSegmentationProcessing.
        """
        logging.info(f"Processing all images in folder: {self.cutout_dir}")
        # Create a dictionary to match images and masks
        # Find all image and mask pairs in the cutout directory
        cropout_image_paths = list(self.cutout_dir.glob("*.jpg"))
        cropout_mask_paths = list(self.cutout_dir.glob("*_mask.png"))

        # Create a mapping from stem (without _mask) to file paths
        cropout_image_paths_map = {img.stem: img for img in cropout_image_paths}
        cropout_mask_paths_map = {mask.stem.replace("_mask", ""): mask for mask in cropout_mask_paths}

        # Process only pairs where both image and mask exist
        for stem in sorted(cropout_image_paths_map.keys() & cropout_mask_paths_map.keys()):
            cropout_image_path = cropout_image_paths_map[stem]
            mask_image_path = cropout_mask_paths_map[stem]
            self.process_single_image(cropout_image_path, mask_image_path)
                    
def main(cfg: DictConfig) -> None:
    """
    Entry point for running the UNet segmentation inference pipeline.

    Args:
        cfg (DictConfig): Configuration with paths to input image directory and trained model.
    """
    post_seg_processor = PostSegmentationProcessing(cfg)
    post_seg_processor.process_cutout_dir()
    logging.info("Post-segmentation processing completed successfully.")