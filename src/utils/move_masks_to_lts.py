import os
import shutil
import pandas as pd

def move_good_images_masks_to_lts(source_dir, image_names_file, dest_dir):
    """
    Move images with tag 'good' from source_dir to dest_dir based on a CSV file.

    Args:
        source_dir (str): Directory containing the source images.
        image_names_file (str): Path to the CSV file with columns [image_name, tag].
        dest_dir (str): Directory to move the selected images to.
    """
    # Load the CSV file
    df = pd.read_csv(image_names_file, header=None, names=["image_name", "tag"])

    # Filter rows where tag == 'good'
    good_images = df[df["tag"] == "good"]["image_name"].tolist()

    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Move each 'good' image to the destination directory
    for image_name in good_images:
        source_image_path = os.path.join(source_dir, image_name)
        dest_image_path = os.path.join(dest_dir, image_name)

        if os.path.exists(source_image_path):
            shutil.move(source_image_path, dest_image_path)
            print(f"Moved {image_name} to {dest_dir}")
        else:
            print(f"Image {image_name} not found in {source_dir}")
