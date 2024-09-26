"""
YOLO Segment Converter for Field Processing Pipeline

This script converts binary segmentation masks into YOLO segmentation format, merging all segments into
a single instance per mask. It processes each mask by merging multiple segments (contours) based on their
spatial proximity, ensuring smooth transitions between segments using interpolation. The resulting YOLO-format
segment is saved as a space-separated text file (Ultralytics YOLO instance segmentation format).

Adapted from https://docs.ultralytics.com/reference/data/converter/

**Note**:
1. **Single Class (1)**: All segments and objects are merged into a single class (class 1), regardless of
   their original labels in the mask. This is suitable for AgIR Field processing pipelines where the goal
   is to treat all objects as a single entity.
   
2. **Long Processing Time for Large Masks**: This script may take a long time to process very large masks,
   especially if the masks contain many segments. The merging process involves finding the nearest points
   between segments, which can be computationally expensive for large numbers of contours.

**Directory Structure**:
    - mask_dir/ : Directory containing the input binary mask images. These masks are grayscale images where 
                  pixel values represent different objects, but all objects are merged into one class.
        Example:
            ├─ mask_image_01.png
            ├─ mask_image_02.png
            └─ ...

    - output_dir/ : Directory where the resulting YOLO segmentation files (in .txt format) will be saved.
        Example:
            ├─ mask_image_01.txt
            ├─ mask_image_02.txt
            └─ ...

**Usage**:
    Initialize the YOLOSegmentConverter class with the mask and output directories, and call the `process_masks`
    method to convert and save YOLO-formatted text files for each mask.

**Example**:
    converter = YOLOSegmentConverter("/path/to/mask_dir", "/path/to/output_dir")
    converter.process_masks()
"""
from pathlib import Path
import numpy as np
import cv2
import logging

from omegaconf import DictConfig

# Configure logging
log = logging.getLogger(__name__)

class YOLOSegmentConverter:
    """
    A class to convert binary segmentation masks into YOLO segmentation format, merge multiple segments,
    and save the output as text files in a specified directory. Processes multiple mask files in parallel.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the YOLOSegmentConverter with the input directories and number of classes.

        Args:
            mask_dir (str): Path to the directory containing binary mask images.
            output_dir (str): Path to the directory where the output will be saved.
            classes (int): Total number of classes in the dataset. Default is 255.
            num_workers (int): Number of parallel workers for multiprocessing. Default is 4.
        """
        log.info("Initializing YOLOSegmentConverter...")
        self.mask_dir = Path(cfg.paths.masks_dir)
        self.output_dir = Path(cfg.paths.yolo_format_label)
        self.classes = cfg.masks_to_yolo.num_classes
        self.num_workers = cfg.masks_to_yolo.num_workers

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def min_index(self, arr1, arr2):
        """
        Find a pair of indexes with the shortest distance between two arrays of 2D points.

        Args:
            arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
            arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

        Returns:
            (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)
    
    def merge_multi_segment(self, segments):
        """
        Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
        This function connects these coordinates with a thin line to merge all segments into one.

        Args:
            segments (List[List]): Original segmentations in COCO's JSON file.
                                Each element is a list of coordinates, like [segmentation1, segmentation2,...].

        Returns:
            s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
        """
        # Reshape each segment into (N, 2) shape
        segments = [np.array(i).reshape(-1, 2) for i in segments]
        # Skip segments that are too small to process
        segments = [seg for seg in segments if len(seg) >= 50]

        # If there's only one valid segment, return it without merging
        if len(segments) == 1:
            log.info("Only one segment found. No merging required.")
            return segments[0]

        # If there are no segments after filtering, return an empty array
        if len(segments) == 0:
            log.warning("No valid segments found. Returning an empty array.")
            return np.array([])
        
        
        s = []
        # segments = [np.array(i).reshape(-1, 2) for i in segments]
        idx_list = [[] for _ in range(len(segments))]

        # Record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self.min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # Use two round to connect all the segments
        for k in range(2):
            # Forward connection
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # Middle segments have two indexes, reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]

                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    # Deal with the first segment and the last one
                    if i in {0, len(idx_list) - 1}:
                        s.append(segments[i])
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0] : idx[1] + 1])

            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    if i not in {0, len(idx_list) - 1}:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return s


    def convert_segment_masks_to_yolo_seg(self, mask_path: Path, classes: int) -> list:
        """
        Converts a segmentation mask to YOLO format by finding contours and normalizing the coordinates.

        **Note**: All segments are merged into a single class (class 0), regardless of their original class.

        Args:
            mask_path (Path): Path to the binary mask image file.

        Returns:
            list: A list of YOLO-formatted segmentation data (class_id, normalized coordinates).
        """
        pixel_to_class_mapping = {i + 1: i for i in range(classes)}
        log.info(f"Converting mask to YOLO format: {mask_path}")
        if mask_path.suffix == ".png":
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask == 255, 0, 1)  # Convert everything into class 0
            img_height, img_width = mask.shape
            unique_values = np.unique(mask)
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue
                class_index = pixel_to_class_mapping.get(value, -1)

                if class_index == -1:
                    log.warning(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # Find contours for the current class
                contours, _ = cv2.findContours((mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        contour = contour.squeeze()
                        yolo_format = [class_index]
                        for point in contour:
                            yolo_format.append(round(point[0] / img_width, 6))
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
        return yolo_format_data



    def convert_and_save(self, data: list, file_path: Path):
        with open(file_path, 'w') as file:

            # for item in data:
            line = " ".join(map(str, data))
            file.write(line + "\n")

        log.info(f"Saved YOLO segmentation file to {file_path}")
    


    def process_single_mask(self, mask_path: Path):
        """Processes a single mask file, converts it to YOLO format, and saves the result (no merging)."""
        log.info(f"Processing single mask: {mask_path}")

        # Convert masks to YOLO format
        results = self.convert_segment_masks_to_yolo_seg(mask_path, classes=255)
        if not results or len(results) == 0:
            log.warning(f"No results found for mask: {mask_path}")
            return
        
        # Extract the segments (excluding the class_id)
        coco_segments = [res[1:] for res in results]  # Only take the coordinates
    
        if len(coco_segments) > 1:
            s = self.merge_multi_segment(coco_segments)
            s = (np.concatenate(s, axis=0)).reshape(-1).tolist()
        else:
            s = [j for i in coco_segments for j in i]  # all segments concatenated
            s = (np.array(s).reshape(-1, 2)).reshape(-1).tolist()
        
        s = [0] + s

        return s


    def process_masks_sequentially(self):
        """Processes all mask files in the input mask directory sequentially."""
        log.info(f"Processing sequentially masks from directory: {self.mask_dir}")
        mask_paths = [mask_path for mask_path in self.mask_dir.iterdir() if mask_path.suffix == '.png']
        mask_paths = [mask_path for mask_path in mask_paths if mask_path.stem not in [x.stem for x in self.output_dir.glob("*.txt")]]
        log.info(f"Processing {len(mask_paths)} masks from directory: {self.mask_dir}")
        for mask_path in mask_paths:
            results  = self.process_single_mask(mask_path)
            if not results:
                log.warning(f"No results found for mask: {mask_path}")
                continue
            output_path = self.output_dir / f"{mask_path.stem}.txt"
            self.convert_and_save(results, output_path)


# Example usage
def main(cfg: DictConfig) -> None:
    converter = YOLOSegmentConverter(cfg)
    converter.process_masks_sequentially()
