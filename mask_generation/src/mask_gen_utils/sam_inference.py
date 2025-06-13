import cv2
import json
import torch
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from segment_anything_hq import sam_model_registry, SamPredictor


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SamSegmenter:
    """
    SamSegmenter is a class for performing segmentation on images using the Segment Anything Model (SAM).

    Attributes:
        device (str): The device to run the model on ('cuda' if available, else 'cpu').
        model: The loaded SAM model.
        predictor: The SAM predictor object for inference.

    Methods:
        __init__(checkpoint: str, model_type: str):
            Initializes the SamSegmenter with a given model checkpoint and type.

        _load_model(checkpoint: str, model_type: str):
            Loads the SAM model from the specified checkpoint and model type.

        yolo_to_xyxy(rel_bbox, image_width, image_height):
            Converts a YOLO-format bounding box (relative coordinates) to absolute pixel coordinates in [x0, y0, x1, y1] format.

        read_bbox_from_txt(bbox_txt_path: str):
            Reads a bounding box from a text file in YOLO format and returns it as a list of floats.

        segment_and_cutout(image_path: str, output_path: str, bbox_txt_path):
            Performs segmentation on the specified image using the bounding box from the text file, and saves the cropped cutout and mask to the output path.

        process_folder(batch_dir: Path):
            Processes all images and corresponding bounding box files in the input folder, performing segmentation and saving results to the output folder.
    """
    def __init__(self, batch_dir:Path, checkpoint_path: Path, model_type: str):
        """
        Initializes the SamSegmenter with a given model checkpoint and model type.

        Args:
            checkpoint (str): Path to the model checkpoint.
            model_type (str): Type of the SAM model (e.g., 'vit_h').
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_dir = batch_dir
        self.image_dir = batch_dir / "developed-images"
        self.cutout_dir = batch_dir / "cutouts"
        self.model = self._load_model(checkpoint_path, model_type)
        self.predictor = SamPredictor(self.model)

    def _load_model(self, checkpoint: str, model_type: str):
        """
        Loads the SAM model using the specified model type and checkpoint.

        Args:
            checkpoint (str): Path to the model checkpoint.
            model_type (str): Type of the SAM model.

        Returns:
            Loaded SAM model on the specified device.
        """
        logging.info(f"Loading SAM model '{model_type}' from checkpoint '{checkpoint}'")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(self.device)
        logging.info("SAM model loaded successfully")
        return sam

    def yolo_to_xyxy(self, rel_bbox, image_width, image_height):
        """
        Converts YOLO-format bounding box (relative values) to pixel coordinates [x0, y0, x1, y1].

        Args:
            rel_bbox (list): Bounding box in YOLO format [x_center, y_center, width, height].
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            list: Bounding box in pixel coordinates.
        """
        x_center, y_center, width, height = rel_bbox
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        x_min = int(max(0, x_center - width / 2))
        y_min = int(max(0, y_center - height / 2))
        x_max = int(min(image_width - 1, x_center + width / 2))
        y_max = int(min(image_height - 1, y_center + height / 2))

        return [x_min, y_min, x_max, y_max]

    def read_bbox_from_json(self, bbox_json_path: str):
        """
        Reads bounding box information from a JSON file.

        Args:
            bbox_json_path (str): Path to the .json file containing the bounding box.

        Returns:
            list: Bounding box as a list of floats [x_center, y_center, width, height].
        """
        with open(str(bbox_json_path), 'r') as f:
            data = json.load(f)
            return data['bbox']
        
    def segment_and_cutout(self, image_path: str, output_path: str, bbox_json_path):
        """
        Performs segmentation using SAM and saves the cropped cutout and mask based on YOLO bounding box.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the cropped cutout.
            bbox_txt_path (str): Path to the YOLO-format bounding box text file.
        """
        logging.info(f"Processing image: {image_path}")
        image = np.array(Image.open(image_path).convert("RGB"))
        image_height, image_width = image.shape[:2]

        rel_bbox = self.read_bbox_from_json(bbox_json_path)
        bbox = self.yolo_to_xyxy(rel_bbox, image_width, image_height)

        self.predictor.set_image(image)

        masks, _, _ = self.predictor.predict(
            box=np.array([bbox]),
            multimask_output=True,
        )

        if masks is None or len(masks) == 0:
            logging.warning(f"No masks predicted for image: {image_path}")
            return

        image = cv2.imread(image_path)
        cropout_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        cropout_image_save_path = str(output_path).replace("_cutout.png", "_cropped.jpg")
        Image.fromarray(cropout_image).save(cropout_image_save_path)
        logging.info(f"Saved cropped image to: {cropout_image_save_path}")

        mask = masks[0].astype(np.uint8)
        mask_path = str(output_path).replace("_cutout.png", "_mask.png")
        Image.fromarray(mask).save(mask_path)
        logging.info(f"Saved mask image to: {mask_path}")

        cutout = cv2.bitwise_and(image, image, mask=mask)
        cutout = cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
        cutout_cropped = cutout[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        Image.fromarray(cutout_cropped).save(output_path)
        logging.info(f"Saved cropped cutout image to: {output_path}")

    def process_folder(self):
        """
        Processes all images and corresponding YOLO-format bounding box files in a folder.

        Args:
            batch_dir (Path): Directory containing images and json bounding box files in sub-directories.
        """
        logging.info(f"Processing dir: {self.batch_dir}")

        images = {img.stem: img for img in self.image_dir.glob("*.jpg")}
        json_files = {json_file.stem: json_file for json_file in self.cutout_dir.glob("*.json")}

        matched_stems = images.keys() & json_files.keys()
        unmatched_images = images.keys() - json_files.keys()

        for stem in matched_stems:
            image_path = images[stem]
            bbox_json_path = json_files[stem]
            output_path = self.cutout_dir / f"{stem}_cutout.png"
            print(f"Found matching bounding box json file for {image_path.name}: {bbox_json_path.name}")
            self.segment_and_cutout(image_path, output_path, bbox_json_path)

        for stem in unmatched_images:
            logging.warning(f"No matching bounding box file for {images[stem].name}")

        logging.info(f"Finished processing folder: {self.batch_dir}")

if __name__ == "__main__":
    checkpoint_path = Path("/home/nsingh27/Field-SegmentationTraining/mask_generation/data/sam_checkpoint/sam_hq_vit_h.pth")
    model_type = "vit_h"
    batch_dir = Path("/home/nsingh27/Field-SegmentationTraining/mask_generation/data/image_processing_dir")

    segmenter = SamSegmenter(batch_dir, checkpoint_path, model_type)
    segmenter.process_folder()