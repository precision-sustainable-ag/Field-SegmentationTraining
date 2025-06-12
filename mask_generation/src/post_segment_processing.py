import cv2
import logging
import numpy as np
from pathlib import Path
import skimage.morphology as morph
from utils_post_seg import make_exg

logging.basicConfig(level=logging.INFO)

class PostSegmentProcessing:
    """
    Class for post-segmentation image processing using ExG thresholding and white pixel filtering.

    This pipeline includes:
    - Loading an image
    - Extracting vegetation (ExG) and white pixel masks
    - Combining and refining masks
    - Applying the final mask to the original image
    - Saving the output
    """

    def __init__(self, cropped_image: Path, mask_image: Path):
        """
        Initialize the processor with the path to the image.

        Args:
            image_path (Path): Path to the input image.
        """
        self.cropped_image = cropped_image
        self.unet_mask_image = mask_image
        self.image = None
        self.exg_mask = None
        self.white_mask = None
        self.gray_mask = None
        self.non_green_stem_mask = None
        self.red_rgb_mask = None

        # Set up output directory
        output_folder = self.cropped_image.parent.parent / "post_segment_processing"
        output_folder.mkdir(parents=True, exist_ok=True)
        self.output_image_path = output_folder / self.cropped_image.name

    def load_image(self):
        """
        Load the image from disk into memory.
        """
        self.image = cv2.imread(str(self.cropped_image))
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.cropped_image}")

    def load_mask(self):
        """
        Load the mask image from disk into memory.
        This method is not currently used but can be implemented if needed.
        """
        self.unet_mask_image = cv2.imread(str(self.unet_mask_image), cv2.IMREAD_GRAYSCALE)
        if self.unet_mask_image is None:
            raise ValueError(f"Failed to load mask image: {self.unet_mask_image}")

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

    def generate_non_green_stem_mask_reddish(self):
        """
        Generate a non-green stem mask from the original image.
        """
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # Lower red
        red_lower1 = (0, 100, 30) # works best: (0, 100, 40)
        red_upper1 = (179, 255, 255) # works best: (179, 255, 255)
        non_green_stem_mask = cv2.inRange(hsv_image, red_lower1, red_upper1)

        return non_green_stem_mask

    def refine_mask(self):
        """
        Apply morphological operations to clean up the combined mask.
        """
        combined_mask = morph.dilation(self.combined_mask)
        combined_mask = morph.opening(self.combined_mask, morph.disk(3))
        combined_mask = morph.closing(self.combined_mask, morph.disk(4))
        combined_mask = morph.erosion(combined_mask, morph.disk(3))
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

    def overlay_mask(self, mask: np.ndarray):
        """
        Overlay the mask on the original image for visualization.

        Args:
            mask (np.ndarray): The binary mask to overlay.
        
        Returns:
            np.ndarray: The image with the mask overlay.
        """
        # Create a red mask where mask > 0
        red_mask = np.zeros_like(self.image)
        red_mask[mask > 0] = [0, 0, 255]  # BGR for red
        return cv2.addWeighted(self.image, 0.7, red_mask, 0.7, 0)
    
    def process_image(self):
        """
        Main method to run the full post-segmentation processing pipeline.
        """
        logging.info(f"Starting post-segmentation processing for: {self.cropped_image}")
        self.load_image()
        self.load_mask()
        # self.exg_mask = self.generate_exg_mask()
        # self.white_mask = self.generate_white_mask()
        # self.gray_mask = self.generate_gray_mask()
        self.non_green_stem_mask = self.generate_non_green_stem_mask_reddish()

        self.combined_mask = cv2.bitwise_or(self.unet_mask_image, self.non_green_stem_mask)
        self.combined_mask = self.refine_mask()
        self.combined_mask = np.where(self.combined_mask > 0, 255, 0).astype(np.uint8) # Convert to binary mask

        # final_cutout_image = cv2.bitwise_and(self.image, self.image, mask=self.combined_mask)
        final_cutout_image = cv2.bitwise_and(self.image, self.image, mask=self.combined_mask)

        mask_output_path = Path(str(self.output_image_path).replace(".jpg", "_mask.png"))

        # Plot the original image and final mask side by side
        self.plot_side_by_side("Original Image", self.image, "Final Mask", final_cutout_image)

        # Overlay the mask on the original image and save it
        overlaid_image = self.overlay_mask(self.combined_mask)
        cv2.imwrite(str(self.output_image_path).replace(".jpg", "_overlaid.jpg"), overlaid_image)

        # Save the final mask
        logging.info(f"Saving final mask to: {mask_output_path}")
        cv2.imwrite(str(mask_output_path), self.combined_mask)

        # Save the final cutout image
        # logging.info(f"Saving final image to: {self.output_image_path}")
        # cv2.imwrite(str(self.output_image_path), final_cutout_image)
        logging.info(f"Post-segmentation processing complete for: {self.cropped_image}")

if __name__ == "__main__":
    input_folder = Path("/home/nsingh27/Field-AnnotationPipeline/data/temp/non_green_stem/cutouts")
    
    # Create a dictionary to match images and masks
    images = {}
    for file in input_folder.glob("*"):
        stem = file.stem.replace("_mask", "")
        if stem not in images:
            images[stem] = {}
        if file.name.endswith(".jpg"):
            images[stem]["image"] = file
        elif file.name.endswith("_mask.png"):
            images[stem]["mask"] = file

    for stem, pair in images.items():
        cropout_image = pair.get("image")
        mask_image = pair.get("mask")
        if not cropout_image or not mask_image:
            logging.warning(f"Skipping incomplete pair: {stem}")
            continue
        
        print(f"Cropout image: {cropout_image}, Mask image: {mask_image}")
        
        post_segment_processor = PostSegmentProcessing(cropout_image, mask_image)
        post_segment_processor.process_image()
    
    logging.info(f"All images processed in folder: {input_folder}.")
