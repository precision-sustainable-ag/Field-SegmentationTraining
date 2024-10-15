import cv2
import numpy as np
import os
def convert_mask(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Create a mask where white (255) becomes black (0) and everything else becomes 1
    converted_image = np.where(image == 255, 0, 1)
    
    # Save the converted image
    cv2.imwrite(output_path, converted_image)

if __name__ == "__main__":
    mask_dir = 'data/masks'
    output_mask_dir = 'data/new_masks'
    os.makedirs(output_mask_dir, exist_ok=True)
    for mask in os.listdir(mask_dir):
        input_path = os.path.join(mask_dir, mask)
        output_path = os.path.join(output_mask_dir, mask)
        convert_mask(input_path, output_path)