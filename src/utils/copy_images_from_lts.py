"""
Script to copy selected images in a text file from 'developed-images' subdirectories in the longer-term storage storage
to a specified destination directory.

Requirements:
- The image names to copy should be listed line-by-line in a text file.
"""

import os
import shutil

def copy_selected_images(source_root, image_names_file, dest_dir):
    """
    Copies images listed in a file from all 'developed-images' subfolders under source_root to dest_dir.

    Args:
        source_root (str): Long-term storage directory containing subfolders with 'developed-images' directories.
        image_names_file (str): Path to the text file listing image names to be copied (one per line).
        dest_dir (str): Path to the destination directory where selected images will be copied.

    Returns:
        None
    """
    # Read image names from the text file
    with open(image_names_file, "r") as f:
        image_names = [line.strip() for line in f if line.strip()]

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through each subfolder inside the source root
    for subfolder in os.listdir(source_root):
        developed_images_path = os.path.join(source_root, subfolder, "developed-images")
        
        # Only proceed if 'developed-images' exists
        if os.path.isdir(developed_images_path):
            for image_name in image_names:
                src_image_path = os.path.join(developed_images_path, image_name)
                
                # If the image exists, copy it to the destination
                if os.path.isfile(src_image_path):
                    dst_image_path = os.path.join(dest_dir, image_name)
                    print(f"Copying {src_image_path} to {dst_image_path}")
                    shutil.copy(src_image_path, dst_image_path)

# Scrip execution
image_names_file = "/home/nsingh27/Field-SegmentationTraining/mask_generation/copy_from_lts.txt"
lts_source_root = "/mnt/research-projects/r/raatwell/longterm_images3/field-batches"
dest_dir = "/home/nsingh27/Field-SegmentationTraining/mask_generation/data/image_processing_dir/developed-images"

copy_selected_images(lts_source_root, image_names_file, dest_dir)