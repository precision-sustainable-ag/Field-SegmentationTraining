import os
import cv2
import numpy as np
from pathlib import Path

def mask_to_yolo_segmentation(mask_path, output_path, class_id=0):
    """
    Converts a binary mask image to YOLO segmentation format and saves the annotations to a file.
    Args:
        mask_path (str): Path to the input binary mask image file.
        output_path (str): Path to the output file where YOLO annotations will be saved.
        class_id (int, optional): Class ID to be used in the YOLO annotations. Defaults to 0.
    Returns:
        None
    The function performs the following steps:
    1. Loads the binary mask image from the specified path.
    2. Converts the mask image to grayscale.
    3. Binarizes the grayscale mask image.
    4. Finds contours (polygons) of the objects in the mask.
    5. Approximates the contours to reduce the number of points (optional).
    6. Prepares YOLO annotations with normalized polygon points.
    7. Saves the YOLO annotations to the specified output file.
    """

    # Load the mask
    mask = cv2.imread(mask_path)

    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Binarize the mask
    binary_mask = np.where(mask == 255, 255, 0).astype(np.uint8) 

    image_height, image_width = binary_mask.shape[:2]

    # Find contours (polygons) of the objects in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_annotations = []

    for contour in contours:
        # Approximate the contour to reduce the number of points (optional)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Prepare YOLO annotation with normalized polygon points
        annotation = f"{class_id}"  # Single class ID
        for point in contour:
            x, y = point[0]
            x_norm = x / image_width
            y_norm = y / image_height
            annotation += f" {x_norm} {y_norm}"
        
        yolo_annotations.append(annotation)

    # Save YOLO annotations to a file
    with open(output_path, 'w') as f:
        for annotation in yolo_annotations:
            f.write(annotation + '\n')

# Set directories
masks_dir = "data/YOLO_dataset/custom_data_fine_tuning/masks"
yolo_format_labels = "data/YOLO_dataset/custom_data_fine_tuning/label_yolo_format"

# Ensure the output directory exists
if not os.path.exists(yolo_format_labels):
    os.makedirs(yolo_format_labels)

for mask in os.listdir(masks_dir):
    mask_path = os.path.join(masks_dir, mask)
    mask_path_stem = Path(mask_path).stem
    yolo_format_path = os.path.join(yolo_format_labels, f"{mask_path_stem}.txt")
    mask_to_yolo_segmentation(mask_path, yolo_format_path)
