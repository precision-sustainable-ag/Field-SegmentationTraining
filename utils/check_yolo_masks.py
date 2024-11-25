import cv2
import os
import numpy as np
from pathlib import Path

def overlay_yolo_masks(image_path, label_path, overlayed_mask_dir, scale_percent=50):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to reduce its size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    height, width, _ = image.shape

    # Read YOLO labels
    with open(label_path, "r") as file:
        labels = file.readlines()

    # Overlay each segmentation mask
    for label in labels:
        # Split the label into class_id and polygon points
        parts = list(map(float, label.strip().split()))
        class_id = int(parts[0])
        bbox_and_polygon = parts[1:]

        # Extract the polygon points
        polygon_points = np.array(bbox_and_polygon[4:]).reshape(-1, 2)
        polygon_points[:, 0] *= width  # Scale x-coordinates
        polygon_points[:, 1] *= height  # Scale y-coordinates
        polygon_points = polygon_points.astype(np.int32)

        # Generate a random color for the mask
        color = tuple(np.random.randint(0, 256, size=3).tolist())

        # Draw the filled polygon on a copy of the image
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon_points], color)

        # Blend the overlay with the original image
        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

        # Draw the polygon outline
        cv2.polylines(image, [polygon_points], isClosed=True, color=color, thickness=2)

        # Label the mask with its class_id
        cv2.putText(image, str(class_id), (polygon_points[0][0], polygon_points[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save the result image
    image_name = image_path.split("/")[-1].split(".")[0]
    output_path = f"{overlayed_mask_dir}/{image_name}_overlayed_yolo_mask.jpg"
    cv2.imwrite(output_path, image)
    print(f"Overlayed image saved at: {output_path}")

# Call the function
overlayed_mask_dir = "del_test_yolo_masks/overlayed_masks"
Path(overlayed_mask_dir).mkdir(parents=True, exist_ok=True)

for image in os.listdir('del_test_yolo_masks/images'):
    image_path = f"del_test_yolo_masks/images/{image}"
    label_path = f"del_test_yolo_masks/labels/{image.split('.')[0]}.txt"
    overlay_yolo_masks(image_path, label_path, overlayed_mask_dir)

