from ultralytics import YOLO
from PIL import Image
import cv2
from pathlib import Path
import torch
import json
from threading import Thread
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import random

def load_metadata(metadata_path: Path):
    """Loads metadata from a JSON file."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata

def resize_or_pad_image_array(image_array: np.ndarray, target_size=(1024, 1024)) -> np.ndarray:
    """
    Resize or pad a NumPy image array to the target size.

    Args:
        image_array (np.ndarray): The input image array (H, W, C).
        target_size (tuple): The target size (width, height) to resize or pad to.

    Returns:
        np.ndarray: The resized or padded image as a NumPy array.
    """
    # Get the original dimensions
    height, width = image_array.shape[:2]
    target_width, target_height = target_size

    # Scale down the image if it is larger than the target size
    if width > target_width or height > target_height:
        # Calculate the aspect ratio
        scale_factor = min(target_width / width, target_height / height)
        new_size = (int(width * scale_factor), int(height * scale_factor))
        image_array = cv2.resize(image_array, new_size, interpolation=cv2.INTER_AREA)

    # Get the new dimensions after scaling
    new_height, new_width = image_array.shape[:2]

    # Calculate padding needed to center the image
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    # Pad the image with black pixels (value [0, 0, 0])
    padded_image = cv2.copyMakeBorder(
        image_array, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded_image

# Load a model
model_path = "../../runs/segment/train/weights/best.pt"

data_root = "../../data/temp"
batches = Path(data_root).glob("*")

images = []
metadata = []
data = []
for batch in batches:
    imgs = Path(batch, "developed-images").glob("*.jpg")
    for img in imgs:
        meta = Path(batch, "cutouts", f"{img.stem}.json")
        data.append((img, meta))
        



random.shuffle(data)
    

output_dir = Path("../../runs/segment/train/results")
output_dir.mkdir(exist_ok=True, parents=True)

multithread = False
imgsz = 1024
retina_masks = True


# Load a pretrained YOLOv8n model
model = YOLO(model_path)

for imgp, metap in data:
# for imgp, metap in zip(images, metadata):
    meta = load_metadata(metap)
    
    bbox = meta["annotation"]["bbox_xywh"]
    
    # for annotation in [meta["annotation"]]:
    image_id = meta["image_info"]["Name"]
        # bbox = annotation["bbox_xywh"]
    img_bgr = cv2.imread(str(imgp))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_cropped = img_rgb[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    img_cropped = resize_or_pad_image_array(img_cropped, target_size=(imgsz, imgsz))
    

    # Run inference on 'bus.jpg' with arguments
    results = model.predict(img_cropped, save=False, imgsz=imgsz, retina_masks=retina_masks)
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # plot on GPU (fast)
        im_rgb = im_bgr[..., ::-1]  # RGB-order PIL image

        cv2.imwrite(f"{output_dir}/{image_id}.jpg", im_rgb)
