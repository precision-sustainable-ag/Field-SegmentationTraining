"""
YOLO Segment Converter for Field Processing Pipeline

This script converts binary segmentation masks into YOLO segmentation format, merging all segments into
a single instance per mask. It processes each mask by merging multiple segments (contours) based on their
spatial proximity, ensuring smooth transitions between segments using interpolation. The resulting YOLO-format
segment is saved as a space-separated text file (Ultralytics YOLO instance segmentation format).

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

import sys
from pathlib import Path
import numpy as np
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from omegaconf import DictConfig
import gc  # Garbage collection

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
        self.mask_dir = Path(cfg.paths.masks_dir)
        self.output_dir = Path(cfg.paths.yolo_format_label)
        self.classes = cfg.masks_to_yolo.num_classes
        self.num_workers = cfg.masks_to_yolo.num_workers

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> tuple:
        """
        Find the index of the closest points between two arrays of 2D points.

        Args:
            arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
            arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

        Returns:
            tuple: A tuple containing the indexes of the closest points in arr1 and arr2 respectively.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def convert_segment_masks_to_yolo_seg(self, mask_path: Path) -> list:
        """
        Converts a segmentation mask to YOLO format by finding contours and normalizing the coordinates.

        **Note**: All segments are merged into a single class (class 0), regardless of their original class.

        Args:
            mask_path (Path): Path to the binary mask image file.

        Returns:
            list: A list of YOLO-formatted segmentation data (class_id, normalized coordinates).
        """
        if mask_path.suffix == ".png":
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask == 255, 0, 1)  # Convert everything into class 0
            img_height, img_width = mask.shape
            unique_values = np.unique(mask)
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue
                class_index = 1 # Always class 1

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

    def merge_multi_segment_with_interpolation(self, segments: list, num_interp_points: int = 10) -> np.ndarray:
        """
        Merges multiple segments by connecting the coordinates of the closest points and interpolates
        a smooth transition between them.

        Args:
            segments (list): A list of segmentation points (in normalized YOLO coordinates).
            num_interp_points (int): Number of interpolated points between segments to smooth transitions.

        Returns:
            np.ndarray: A merged NumPy array of concatenated points in normalized coordinates.
        """
        segments = [np.array(i).reshape(-1, 2) for i in segments]

        if not segments:
            return np.array([])

        merged_segment = segments.pop(0)

        while segments:
            last_point_in_merged = merged_segment[-1]
            min_dist = float('inf')
            closest_segment = None
            closest_idx = None
            closest_point_in_merged = None
            closest_point_in_current = None

            for i, segment in enumerate(segments):
                idx1, idx2 = self.min_index(merged_segment, segment)
                dist = np.linalg.norm(merged_segment[idx1] - segment[idx2])

                if dist < min_dist:
                    min_dist = dist
                    closest_segment = segment
                    closest_idx = i
                    closest_point_in_merged = idx1
                    closest_point_in_current = idx2

            transition_points = self.interpolate_points(merged_segment[closest_point_in_merged],
                                                        closest_segment[closest_point_in_current],
                                                        num_interp_points)

            closest_segment = np.roll(closest_segment, -closest_point_in_current, axis=0)
            merged_segment = np.roll(merged_segment, -(closest_point_in_merged + 1), axis=0)

            merged_segment = np.concatenate([merged_segment, transition_points, closest_segment])
            segments.pop(closest_idx)

        return merged_segment

    def interpolate_points(self, p1: np.ndarray, p2: np.ndarray, num_points: int = 10) -> np.ndarray:
        """
        Interpolates a set of points between two points p1 and p2.

        Args:
            p1 (np.ndarray): The first point (2D normalized coordinates).
            p2 (np.ndarray): The second point (2D normalized coordinates).
            num_points (int): Number of interpolated points between p1 and p2.

        Returns:
            np.ndarray: An array of interpolated points between p1 and p2.
        """
        return np.linspace(p1, p2, num_points)

    def convert_and_save(self, data: list, file_path: Path):
        """
        Converts the segmentation data into a flat list and saves it as a text file.

        Args:
            data (list): List of YOLO segmentation data (class_id, normalized coordinates).
            file_path (Path): Path to the output text file.
        """
        result = []
        for item in data:
            if isinstance(item, np.ndarray):
                result.extend(item.tolist())
            else:
                result.append(item)

        with open(file_path, 'w') as f:
            f.write(' '.join(map(str, result)))
        log.info(f"Saved YOLO segmentation file to {file_path}")

    def process_single_mask(self, mask_path: Path):
        """
        Processes a single mask file, converts it to YOLO format, merges the segments, and saves the result.

        Args:
            mask_path (Path): Path to the binary mask image file.
        """
        # Convert masks to YOLO format
        results = self.convert_segment_masks_to_yolo_seg(mask_path)
        if not results:
            log.warning(f"No results found for mask: {mask_path}")
            return

        coco_segments = []
        for res in results:
            coco_segments.append(res[1:])  # Collect only the normalized points from YOLO data

        # Merge segments with interpolation
        merged_segments = self.merge_multi_segment_with_interpolation(coco_segments)
        if merged_segments.size == 0:
            log.warning(f"No merged segments found for mask: {mask_path}")
            return

        # Prepend the class ID to the merged segments
        merged_yolo_format = [1] + merged_segments.flatten().tolist()  # Always class 1

        # Save result to output directory
        output_path = self.output_dir / f"{mask_path.stem}.txt"
        self.convert_and_save(merged_yolo_format, output_path)
        # Free up memory after processing each mask
        gc.collect()


    def process_masks_concurrently(self):
        """Processes all mask files in the input mask directory using multithreading (or multiprocessing)."""
        mask_paths = [mask_path for mask_path in self.mask_dir.iterdir() if mask_path.suffix == '.png']
        mask_paths = [mask_path for mask_path in mask_paths if mask_path.stem not in [x.stem for x in self.output_dir.glob("*.txt")]]

        log.info(f"Processing {len(mask_paths)} masks from directory: {self.mask_dir}")

        # Use ThreadPoolExecutor for multithreading, or ProcessPoolExecutor for multiprocessing
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Use map instead of submit for cleaner concurrency
            list(executor.map(self.process_single_mask, mask_paths))

        log.info("Mask processing complete.")
        gc.collect()

    def process_masks_sequentially(self):
        mask_paths = [mask_path for mask_path in self.mask_dir.iterdir() if mask_path.suffix == '.png']
        mask_paths = [mask_path for mask_path in mask_paths if mask_path.stem not in [x.stem for x in self.output_dir.glob("*.txt")]]
        log.info(f"Processing {len(mask_paths)} masks from directory: {self.mask_dir}")
        for mask_path in mask_paths:
            self.process_single_mask(mask_path)




# Example usage
def main(cfg: DictConfig) -> None:
    converter = YOLOSegmentConverter(cfg)
    converter.process_masks_concurrently()
    # converter.process_masks_sequentially()
