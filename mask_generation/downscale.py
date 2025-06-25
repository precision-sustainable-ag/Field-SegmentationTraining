import cv2
import numpy as np
from pathlib import Path

def downscale_image_keep_size(img, scale_factor):
    # Downscale the image
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Upscale back to original size
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
    return upscaled



img_dir  = Path("data/mask_generation_dir/developed-images")

for img_path in img_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    downscaled_img = downscale_image_keep_size(img, 0.01)
    new_filename = img_path.stem + ".downscaled.jpg"
    img_path = img_path.with_name(new_filename)
    print(f"Saving downscaled image to: {img_path}")
    cv2.imwrite(str(img_path), downscaled_img)