from omegaconf import DictConfig
from PIL import Image
from pathlib import Path
import logging
import fiftyone as fo
import numpy as np
from mask_gen_utils.inspect import FiftyOneMaskInspector

log = logging.getLogger(__name__)

DB_PATH = "/mnt/research-projects/r/raatwell/longterm_images3/field-tools/agir_field_db/test_agir_field.db"
ANNO_KEY = "cvat_mask_refine_run_001"
FIFTYONE_DATASET = "your_mask_dataset"
LABEL_FIELD = "refined_mask"
TABLE_NAME = "mask_gen_images"


def save_combined_detection_masks(returned_dict, dataset, output_mask_dir="temp_masks", mask_field="refined_mask"):
    """
    Combines all detection masks for each image and saves as a PNG mask.

    Args:
        returned_dict: Output from FiftyOne's load_annotations with `unexpected="return"`
        dataset: The FiftyOne Dataset
        output_mask_dir: Where to save PNG masks
        mask_field: The field containing detections ("refined_mask")
    """
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    n_exported = 0

    if returned_dict and mask_field in returned_dict and "detections" in returned_dict[mask_field]:
        detections_dict = returned_dict[mask_field]["detections"]
        for sample_id, det_dict in detections_dict.items():
            sample = dataset[sample_id]
            # Get image size
            if sample.has_field("metadata") and hasattr(sample.metadata, "height") and hasattr(sample.metadata, "width"):
                img_h, img_w = sample.metadata.height, sample.metadata.width
            elif "initial_mask" in sample.field_names and sample["initial_mask"] is not None:
                img_h, img_w = sample["initial_mask"].mask.shape
            else:
                log.warning(f"Cannot determine image shape for {sample.filepath}, skipping.")
                continue

            combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for detection in det_dict.values():
                det_mask = detection.mask
                if det_mask is None:
                    continue
                bbox = detection.bounding_box  # [x0, y0, w, h], relative (0-1)
                x0, y0, w, h = bbox
                x0_pix = int(round(x0 * img_w))
                y0_pix = int(round(y0 * img_w))
                w_pix = int(round(w * img_w))
                h_pix = int(round(h * img_h))
                det_mask_bin = (det_mask > 0).astype(np.uint8)
                try:
                    combined_mask[y0_pix:y0_pix + h_pix, x0_pix:x0_pix + w_pix] |= det_mask_bin[:h_pix, :w_pix]
                except Exception as e:
                    log.warning(
                        f"Error pasting mask for {sample.filepath}, bbox {bbox}, mask shape {det_mask.shape}: {e}"
                    )
            out_path = output_mask_dir / f"{Path(sample.filepath).stem}_mask.png"
            Image.fromarray(combined_mask * 255).save(out_path)
            log.info(f"Saved combined mask for {sample.filepath}")
            n_exported += 1
    log.info(f"✅ Exported {n_exported} combined masks to: {output_mask_dir}")

def main(cfg: DictConfig) -> None:
    inspector = FiftyOneMaskInspector(cfg)
    samples = inspector._load_samples()
    dataset: fo.Dataset = inspector._create_dataset(samples)

    view = dataset.take(3)
    view.annotate(
        ANNO_KEY,
        label_field=LABEL_FIELD,
        label_type="segmentation",
        media_field="filepath",
        mask_targets={0: "background", 255: "weed"},
        launch_editor=True,
    )

    print("\nGo annotate in the CVAT UI. When done, come back here and press ENTER.")
    input("Press ENTER to continue...")

    print(f"Annotation info: {dataset.get_annotation_info(ANNO_KEY)}")
    print("Loading annotations from CVAT...")

    returned_dict = dataset.load_annotations(
        ANNO_KEY,
        # dest_field=LABEL_FIELD,
        unexpected="return"
    )
    save_combined_detection_masks(returned_dict, dataset, output_mask_dir="temp_masks", mask_field=LABEL_FIELD)

    print("Deleting annotation run from FiftyOne...")
    dataset.delete_annotation_run(ANNO_KEY)
    print("✅ Annotations loaded, combined, and exported!")

