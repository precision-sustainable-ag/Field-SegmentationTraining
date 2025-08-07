import os
from PIL import Image

# === CONFIGURATION ===
input_folder = "/home/sbaghba/envs/Field-SegmentationTraining/projects/test_project/preprocess/train_val_test_split/train/masks"
output_folder = "/home/sbaghba/envs/Field-SegmentationTraining/projects/test_project/preprocess/train_val_test_split/train/masks1"
target_size = (512, 512)

# === Ensure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

# === Process each image ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        input_path = os.path.join(input_folder, filename)

        # Split name and extension
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_test_resized{ext}"
        output_path = os.path.join(output_folder, new_filename)

        with Image.open(input_path) as img:
            resized_img = img.resize(target_size)
            resized_img.save(output_path)

print("Resizing complete. Saved to:", output_folder)
