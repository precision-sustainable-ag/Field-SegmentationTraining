import os
import numpy as np
import logging
from PIL import Image
from omegaconf import DictConfig
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

def crop_pad_or_resize(image_path, mask_path, output_image_path, output_mask_path, target_size=(1024, 1024)):
    try:
        # Load image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Convert mask to NumPy array
        mask_np = np.array(mask)

        # Get the coordinates of the bounding box where mask != 255
        non_255_coords = np.where(mask_np != 255)
        if non_255_coords[0].size == 0 or non_255_coords[1].size == 0:
            print(f"Skipping {image_path} and {mask_path} as there are no non-255 pixels.")
            return

        top_left_y, top_left_x = np.min(non_255_coords[0]), np.min(non_255_coords[1])
        bottom_right_y, bottom_right_x = np.max(non_255_coords[0]), np.max(non_255_coords[1])

        # Crop the image and mask based on the bounding box
        cropped_image = image.crop((top_left_x, top_left_y, bottom_right_x + 1, bottom_right_y + 1))
        cropped_mask = mask.crop((top_left_x, top_left_y, bottom_right_x + 1, bottom_right_y + 1))

        cropped_width, cropped_height = cropped_image.size

        # If the cropped area is larger than 1024x1024, resize it to target size
        if cropped_width > target_size[0] or cropped_height > target_size[1]:
            resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
            resized_mask = cropped_mask.resize(target_size, Image.Resampling.NEAREST)
            resized_image.save(output_image_path)
            resized_mask.save(output_mask_path)
        else:
            # If the cropped area is smaller than 1024x1024, extend or pad the area
            if cropped_width < target_size[0]:
                extra_width = target_size[0] - cropped_width
                left_padding = extra_width // 2
                right_padding = extra_width - left_padding
                top_left_x = max(0, top_left_x - left_padding)
                bottom_right_x = min(image.width - 1, bottom_right_x + right_padding)

            if cropped_height < target_size[1]:
                extra_height = target_size[1] - cropped_height
                top_padding = extra_height // 2
                bottom_padding = extra_height - top_padding
                top_left_y = max(0, top_left_y - top_padding)
                bottom_right_y = min(image.height - 1, bottom_right_y + bottom_padding)

            # After extending, crop again with new dimensions
            extended_image = image.crop((top_left_x, top_left_y, bottom_right_x + 1, bottom_right_y + 1))
            extended_mask = mask.crop((top_left_x, top_left_y, bottom_right_x + 1, bottom_right_y + 1))

            # If the extended area is still smaller, pad it with black (for image) and white (for mask)
            extended_width, extended_height = extended_image.size
            if extended_width < target_size[0] or extended_height < target_size[1]:
                padded_image = Image.new('RGB', target_size, (0, 0, 0))  # Black padding for the image
                padded_mask = Image.new('L', target_size, 255)  # White padding for the mask

                # Calculate where to place the extended image/mask on the canvas
                offset_x = (target_size[0] - extended_width) // 2
                offset_y = (target_size[1] - extended_height) // 2

                padded_image.paste(extended_image, (offset_x, offset_y))
                padded_mask.paste(extended_mask, (offset_x, offset_y))

                # Save the padded images
                padded_image.save(output_image_path)
                padded_mask.save(output_mask_path)
            else:
                # Save the extended images if no padding is necessary
                extended_image.save(output_image_path)
                extended_mask.save(output_mask_path)
    except Exception as e:
        print(f"Error processing {image_path} and {mask_path}: {e}")


def process_directory(image_dir, mask_dir, output_image_dir, output_mask_dir, target_size=(1024, 1024), max_workers=4):
    # Create output directories if they do not exist
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # Get list of image and mask files
    image_files = sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpeg'))
    mask_files = sorted(mask_dir.glob('*.png'))

    # Ensure there are equal numbers of image and mask files
    if len(image_files) != len(mask_files):
        print("Error: The number of images and masks do not match.")
        return

    # Use ThreadPoolExecutor to parallelize processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_file, mask_file in zip(image_files, mask_files):
            # Define output paths
            output_image_path = output_image_dir / image_file.name
            output_mask_path = output_mask_dir / mask_file.name

            # Submit tasks to executor
            futures.append(executor.submit(crop_pad_or_resize, image_file, mask_file, output_image_path, output_mask_path, target_size))

        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()


def main(cfg: DictConfig) -> None:
    # Example usage
    image_dir = Path(cfg.paths.image_dir)
    mask_dir = Path(cfg.paths.masks_dir)
    output_image_dir = Path(cfg.paths.data_dir, 'cropped_resized_images')
    output_mask_dir = Path(cfg.paths.data_dir, 'cropped_resized_masks')
    target_size = (cfg.crop_and_resize.target_size, cfg.crop_and_resize.target_size)

    log.info("Cropping and resizing images and masks.")
    process_directory(image_dir, mask_dir, output_image_dir, output_mask_dir, target_size=target_size)