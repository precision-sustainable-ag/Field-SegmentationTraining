"""
Script to create and visualize a FiftyOne dataset for image segmentation tasks.

This script:
- Loads image and corresponding mask pairs ground truth and predictions
- Creates FiftyOne samples with segmentation masks
- Opens the FiftyOne App for interactive sample selection
- Saves the names of selected samples to a text file
"""
import fiftyone as fo
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime

def load_samples(image_dir, ground_truth_mask_dir, prediction_mask_dir):
    """
    Loads image and corresponding mask files into FiftyOne samples.

    Args:
        image_dir (str or Path): Directory containing .jpg images.
        ground_truth_mask_dir (str or Path): Directory containing ground truth .png masks.
        prediction_mask_dir (str or Path, optional): Directory containing prediction .png masks.

    Returns:
        List[fo.Sample]: List of FiftyOne samples with segmentation masks.
    """
    samples = []

    for image_path in Path(image_dir).glob("*.jpg"):
        stem = image_path.stem
        ground_truth_mask_path = Path(ground_truth_mask_dir) / f"{stem}.png"

        if not ground_truth_mask_path.exists():
            print(f"Warning: Ground truth mask not found for {image_path.name}. Skipping.")
            continue

        ground_truth_mask_array = np.array(Image.open(ground_truth_mask_path).convert("L"), dtype=np.uint8)

        sample = fo.Sample(filepath=str(image_path))
        sample["ground_truth"] = fo.Segmentation(mask=ground_truth_mask_array)

        if prediction_mask_dir:
            prediction_mask_path = Path(prediction_mask_dir) / f"{stem}.png"
            if prediction_mask_path.exists():
                prediction_mask_array = np.array(Image.open(prediction_mask_path).convert("L"), dtype=np.uint8)
                sample["prediction"] = fo.Segmentation(mask=prediction_mask_array)
            else:
                print(f"Note: Prediction not found for {image_path.name}.")

        samples.append(sample)

    return samples

def get_or_create_dataset(dataset_name, samples):
    """
    Loads an existing FiftyOne dataset or creates a new one.

    Args:
        dataset_name (str): Name of the dataset to load or create.
        samples (List[fo.Sample]): List of samples to add to the dataset.

    Returns:
        fo.Dataset: The loaded or newly created FiftyOne dataset.
    """
    if fo.dataset_exists(dataset_name):
        print(f"Loading existing dataset: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
        if len(dataset) > 0:
            print("Dataset already contains samples. Skipping adding new samples.")
            samples = []
    else:
        print(f"Creating new dataset: {dataset_name}")
        dataset = fo.Dataset(dataset_name)

    dataset.add_samples(samples)
    return dataset

def save_selected_samples(dataset, session, output_path):
    """
    Saves the filenames of user-selected samples in the FiftyOne app to a text file.

    Args:
        dataset (fo.Dataset): The FiftyOne dataset being visualized.
        session (fo.AppSession): The active FiftyOne session.
        output_path (str or Path): File path to write selected sample filenames.
    """
    selected_ids = session.selected
    selected_samples = dataset.select(selected_ids)

    with open(output_path, "w") as f:
        for sample in selected_samples:
            f.write(f"{Path(sample.filepath).name}\n")

    print(f"Selected samples written to {output_path}")

def create_fiftyone_dataset(image_dir, ground_truth_mask_dir, prediction_mask_dir, dataset_name, port):
    """
    Main pipeline to create a FiftyOne dataset and launch the app for visualization.

    Args:
        image_dir (str or Path): Path to the image directory.
        ground_truth_mask_dir (str or Path): Path to the ground truth mask directory.
        prediction_mask_dir (str or Path): Path to the prediction mask directory.
        dataset_name (str): Name of the FiftyOne dataset.
        port (int): Port number to run the FiftyOne app on.
    """
    samples = load_samples(image_dir, ground_truth_mask_dir, prediction_mask_dir)
    dataset = get_or_create_dataset(dataset_name, samples)

    session = fo.launch_app(dataset, port=port)

    try:
        print("Select samples in the FiftyOne app and then close it to continue...")
        print("Press Ctrl+C to interrupt the session manually if needed.")
        session.wait()
    except KeyboardInterrupt:
        print("\nSession manually interrupted by user.")
    finally:
        session.close()
        print("Session closed.")

    if session.selected:
        output_file = Path(image_dir).parent / "selected_samples_with_voxel51.txt"
        save_selected_samples(dataset, session, output_file)
    else:
        print("No samples were selected. No file written.")

if __name__ == "__main__":
    image_dir = Path("/home/nsingh27/Field-SegmentationTraining/data/IMP_non_green_stem_issue/test_set_non_green_stem/image_cropout")
    ground_truth_mask_dir = Path("/home/nsingh27/Field-SegmentationTraining/data/IMP_non_green_stem_issue/test_set_non_green_stem/ground_truth")
    prediction_mask_dir = Path("/home/nsingh27/Field-SegmentationTraining/data/IMP_non_green_stem_issue/test_set_non_green_stem/predicted_mask")  # <-- Set your secondary mask path here
    dataset_name = "IMP_non_green_stem_issue6"
    port = 5152

    create_fiftyone_dataset(
        image_dir,
        ground_truth_mask_dir,
        prediction_mask_dir,
        dataset_name,
        port
    )
