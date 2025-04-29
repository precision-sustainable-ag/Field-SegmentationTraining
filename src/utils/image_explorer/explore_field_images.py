import cv2
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExploreImage:
    def __init__(self, image_dir: Path):
        """
        Initialize the ExploreImage class with the directory containing images.

        Args:
            image_dir (Path): Path to the directory containing images.
        """
        self.image_dir = Path(image_dir)
        self.selected_images = []

    def select_images_with_required_traits(self):
        """
        Allow user to manually select images by viewing them one by one.
        Press '1' to select, '0' to skip, or 'q' to quit early.
        """
        print("\n\nPress '1' to select, '0' to skip, 'q' to quit early.\n\nPlease press enter to continue...")
        input()
        logging.info("Starting image selection process...")

        image_files = [file for file in self.image_dir.iterdir() if file.suffix.lower() in {".jpg", ".png", ".jpeg"}]

        if not image_files:
            logging.warning("No image files found in the directory.")
            return

        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                logging.warning(f"Could not read image: {image_path.name}")
                continue

            # Resize image if too large
            max_width = 800
            if image.shape[1] > max_width:
                scale = max_width / image.shape[1]
                image = cv2.resize(image, (max_width, int(image.shape[0] * scale)))

            cv2.imshow("Image Viewer", image)
            logging.info(f"Viewing: {image_path.name}")

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('1'):
                    self.selected_images.append(image_path)
                    logging.info(f"Selected: {image_path.name}")
                    break
                elif key == ord('0'):
                    logging.info(f"Skipped: {image_path.name}")
                    break
                elif key == ord('q'):
                    logging.info("Quitting early...")
                    cv2.destroyAllWindows()
                    self.save_selected_image_names_to_file()
                    return

        cv2.destroyAllWindows()
        self.save_selected_image_names_to_file()

    def save_selected_image_names_to_file(self):
        """
        Save selected image names to a text file and optionally copy them to a new directory.
        """
        if not self.selected_images:
            logging.info("No images were selected.")
            return

        # Save filenames
        output_file = self.image_dir / "selected_images.txt"
        with open(output_file, "w") as f:
            for img_path in self.selected_images:
                f.write(f"{img_path.name}\n")
        logging.info(f"Saved selected image list to {output_file}")

def main():
    image_dir = Path("/home/nsingh27/Field-SegmentationTraining/data/images_with_flash_issues")  # Set your image directory here
    explorer = ExploreImage(image_dir)
    explorer.select_images_with_required_traits()

if __name__ == "__main__":
    main()
