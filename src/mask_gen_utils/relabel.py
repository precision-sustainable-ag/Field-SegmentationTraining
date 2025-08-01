import os
import cv2
import logging
import datetime
from pathlib import Path
from typing import List, Tuple

import fiftyone as fo
import numpy as np
from PIL import Image
from omegaconf import DictConfig

from src.utils.utils import read_yaml
from src.mask_gen_utils.inspect import FiftyOneMaskInspector

log = logging.getLogger(__name__)


class MaskRelabelPipeline:
    def __init__(self, cfg: DictConfig):
        """Pipeline for mask relabeling and annotation correction in image datasets."""
        self.cfg = cfg
        self.annot_session_key = cfg.mask_gen.relabel.annot_session_key
        self.dest_field = "relabeled_mask"
        self.output_mask_dir = Path(cfg.paths.relabeled_masks_dir)
        self.refined_mask_dir = Path(cfg.paths.refined_masks_dir)
        self.timestamp = datetime.datetime.now().isoformat()
        self.reviewer = os.getenv("USER", "unknown_user")
        self.inspector = FiftyOneMaskInspector(cfg)

        self.keys = read_yaml(cfg.paths.keys_path)
        self.task_name = cfg.mask_gen.relabel.task_name or "mask_relabeling"

    def load_samples_for_relabel(self) -> List[fo.Sample]:
        """
        Load image samples that require relabeling from the database.

        Returns:
            List[fo.Sample]: List of FiftyOne Sample objects, each with paths and tags
            pre-populated. Masks are loaded and attached to each sample.
        """
        rows = self.inspector.db.get_images_for_relabel(
            only_tags=self.cfg.mask_gen.relabel.only_tags
        )
        
        # Iterate through the rows and create FiftyOne samples
        samples = []
        for row in rows:
            (
                image_id, image_path, mask_path, refined_mask_path,
                initial_tag, final_tag, tags, status, reviewer, timestamp,
                refine_params_str
            ) = row
            if not Path(image_path).exists() or not Path(mask_path).exists():
                continue
            
            # Load the initial mask and create a sample
            sample = fo.Sample(filepath=image_path)
            sample["image_id"] = image_id
            sample["mask_path"] = mask_path
            sample["refined_mask_path"] = refined_mask_path
            sample["relabeled_mask_path"] = str(self.output_mask_dir / f"{Path(image_id).stem}_mask.png")
            sample["initial_tag"], sample["final_tag"] = initial_tag, final_tag or ""
            sample["status"], sample["reviewer"] = status, reviewer
            sample["timestamp"], sample["refine_params"] = timestamp, refine_params_str
            sample["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

            sample = self._load_masks(sample)
            samples.append(sample)

        return samples
    
    def _load_masks(self, sample: fo.Sample) -> fo.Sample:
        """
        Attach the initial, refined, and relabeled masks to the sample. 

        Args:
            sample (fo.Sample): Sample object with mask filepaths set.

        Returns:
            fo.Sample: Sample with mask arrays loaded as FiftyOne Segmentation fields.
        """
        file_path = sample['filepath']
        initial_mask_path = Path(sample["mask_path"])
        refined_mask_path = Path(sample["refined_mask_path"])
        relabeled_mask_path = Path(sample["relabeled_mask_path"])
        
        # Load the initial mask
        if initial_mask_path.exists():
            initial_mask_array = np.array(Image.open(initial_mask_path).convert("L"), dtype=np.uint8)
        else:
            log.warning(f"Initial mask not found at {initial_mask_path}. Creating empty mask for relabeling from scratch.")
            image_array = np.array(Image.open(file_path).convert("L"), dtype=np.uint8)
            img_h, img_w = image_array.shape[:2]
            initial_mask_array = np.zeros((img_h, img_w), dtype=np.uint8)
        sample["initial_mask"] = fo.Segmentation(mask=initial_mask_array)

        # Load refined mask if it exists
        if refined_mask_path.exists():
            refined_mask_array = np.array(Image.open(refined_mask_path).convert("L"), dtype=np.uint8)
        else:
            log.warning(f"Refined mask not found at {refined_mask_path}. Using initial mask for relabeling.")
            refined_mask_array = initial_mask_array.copy()
        sample["refined_mask"] = fo.Segmentation(mask=refined_mask_array)

        # Load relabeled mask if it exists else use refined mask
        if relabeled_mask_path.exists():
            relabeled_mask_array = np.array(Image.open(relabeled_mask_path).convert("L"), dtype=np.uint8)
        else:
            relabeled_mask_array = refined_mask_array
        sample["relabeled_mask"] = fo.Segmentation(mask=relabeled_mask_array)

        return sample

    def launch_annotation(self, view: fo.DatasetView) -> None:
        """
        Launch a CVAT annotation session with a segmentation schema for relabeling.

        Args:
            view (fo.DatasetView): The FiftyOne view containing samples for annotation.
        """
        # Set up the segmentation schema for annotation
        segmentation_label_schema = {
            self.dest_field: {
                "label_type": "Segmentation",
                "classes": ["weed"],
                "mask_targets": {255: "weed"},
            }
        }
        view.annotate(
            self.annot_session_key,
            backend="cvat",
            label_schema=segmentation_label_schema,
            launch_editor=False,
            username=self.keys['cvat']['username'],
            password=self.keys['cvat']['password'],
            task_name=self.task_name
        )
        print("\nGo annotate in the CVAT UI. When done, come back here.")

    def import_annotations(self, view: fo.DatasetView) -> None:
        """
        Import relabeled masks from the CVAT annotation session.

        Args:
            view (fo.DatasetView): The FiftyOne view containing samples to update.
        """
        view.load_annotations(
            self.annot_session_key,
            dest_field=self.dest_field,
            unexpected="keep",
        )

    def postprocess_annotations(self, view: fo.DatasetView) -> None:
        """
        Post-process annotated samples by combining instance masks (Detections) into a full mask.
        Cleans up the sample's detection fields after processing.

        Args:
            view (fo.DatasetView): The FiftyOne view containing annotated samples.
        """
        for sample in view:
            full_mask = self.combine_detection_masks_to_full_mask(sample)
            sample["relabeled_mask"] = fo.Segmentation(mask=full_mask)
            if "detections" in sample:
                del sample["detections"]

    def finalize(self, view: fo.DatasetView, save_to_local: bool = False) -> None:
        """
        Finalize the pipeline: save relabeled masks, update database records, and commit changes.

        Args:
            view (fo.DatasetView): The FiftyOne view of processed samples.
        """
        updates = []
        for sample in view:
            status, timestamp, reviewer = self._evaluate_and_update_status(sample)
            
            if save_to_local:
                self._save_or_remove_relabeled_mask_if_needed(sample, status)
            updates.append(self.prepare_update_tuple(sample, status, timestamp, reviewer))
        self._commit_updates(updates)

    @staticmethod
    def same_masks(mask1: np.ndarray, mask2: np.ndarray) -> bool:
        """
        Test if two masks are pixel-identical.
        Returns:
            bool: True if masks are identical, False otherwise.
        """
        if mask1 is None or mask2 is None:
            log.warning("One of the masks is None, cannot compare.")
            return False
        return np.array_equal(mask1, mask2)
    
    @staticmethod
    def combine_detection_masks_to_full_mask(
        sample: fo.Sample,
        detections_field: str ="detections",
        mask_shape_field: str ="refined_mask",
        scale_mask: int = 255
    ) -> np.ndarray:
        """
        Combine all instance detection masks into a full-size mask.

        Args:
            sample (fo.Sample): The sample with detection instances.
            detections_field (str): Name of the detections field.
            mask_shape_field (str): Name of the mask shape field for dimensions.
            scale_mask (int): Value to scale the mask (e.g., 255 for uint8).

        Returns:
            np.ndarray: Full-size mask array.
        """
        if detections_field not in sample or mask_shape_field not in sample:
            return sample[mask_shape_field].mask

        relabeled_detections = sample[detections_field]["detections"]
        img_height, img_width = sample[mask_shape_field].mask.shape[:2]
        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for detection in relabeled_detections:
            detection_mask = detection["mask"]
            mask_h, mask_w = detection_mask.shape
            x, y, _, _ = detection["bounding_box"]
            x1 = int(round(x * img_width))
            y1 = int(round(y * img_height))
            x2 = min(x1 + mask_w, img_width)
            y2 = min(y1 + mask_h, img_height)
            mask_w_clip = x2 - x1
            mask_h_clip = y2 - y1
            full_mask[y1:y2, x1:x2] = detection_mask[:mask_h_clip, :mask_w_clip] * scale_mask

        return full_mask
    
    @staticmethod
    def prepare_update_tuple(sample: fo.Sample, status: str, timestamp: datetime, reviewer: str) -> Tuple:
        """
        Build the tuple for updating the database with sample status and metadata.
        Args:
            sample (fo.Sample): Sample to summarize.
            status (str): The status string to record.
            timestamp (str): ISO timestamp for the update.
            reviewer (str): Reviewer identifier.

        Returns:
            Tuple: Tuple of update fields for database operations.
        """
        image_name = Path(sample.filepath).name
        initial_tag = sample["initial_tag"]
        final_tag = sample["final_tag"]
        refine_params = sample["refine_params"] or "{}"
        tags = ",".join(sorted(list(sample['tags'])))
        return (initial_tag, final_tag, tags, status, reviewer, timestamp, refine_params, image_name)
    
    def _evaluate_and_update_status(self, sample: fo.Sample) -> Tuple[str, str, str]:
        """
        Compare refined and relabeled masks and return the update status for DB.

        Args:
            sample (fo.Sample): The sample to evaluate.

        Returns:
            Tuple[str, str, str]: (status, timestamp, reviewer)
        """
        refined_mask = sample["refined_mask"].mask
        relabeled_mask = sample["relabeled_mask"].mask
        are_same = self.same_masks(refined_mask, relabeled_mask)

        if are_same:
            # No change needed
            return sample["status"], sample["timestamp"], sample["reviewer"]
        else:
            # Updated by relabeling
            return "relabelled", self.timestamp, self.reviewer

    def _save_or_remove_relabeled_mask_if_needed(self, sample: fo.Sample, status: str) -> None:
        """
        Save the relabeled mask as a PNG if it was updated; remove otherwise.

        Args:
            sample (fo.Sample): The sample containing the relabeled mask.
            status (str): The current status (determines if a save is needed).
        """
        if status == "relabelled":
            relabeled_mask = sample["relabeled_mask"].mask
            image_name = Path(sample.filepath).name
            self.output_mask_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_mask_dir / image_name.replace(".jpg", "_mask.png")
            cv2.imwrite(str(output_path), relabeled_mask)
            log.info(f"Saved relabeled mask: {output_path}")
        else:
            del sample["relabeled_mask"]

    def _commit_updates(self, updates: List[Tuple]) -> None:
        """Bulk update DB and commit."""
        self.inspector.db.bulk_update_tags(updates)
        self.inspector.db.commit()

    def run(self) -> None:
        """
        Execute the full relabeling pipeline:
          1. Load samples
          2. Launch annotation
          3. Import annotation (CVAT -> FiftyOne)
          4. Post-process mask fields
          5. Finalize and save results
        """
        try:
            samples = self.load_samples_for_relabel()
            dataset = self.inspector.create_dataset(samples)
            dataset.save()

            if len(dataset) == 0:
                print("No images needing relabel found in the database.")
                return

            view = dataset  # Could be dataset.take(N) if sampling desired

            self.launch_annotation(view)
            input("Press ENTER to continue after annotating in CVAT...")
    
        except Exception as e:
            log.error(f"Error during relabeling pipeline: {e}")
    
        finally:
            log.info("Finalizing relabeling pipeline...")
            self.import_annotations(view)
            self.postprocess_annotations(view)
            self.finalize(view, export_from_cvat=False)


def main(cfg: DictConfig) -> None:
    pipeline = MaskRelabelPipeline(cfg)
    pipeline.run()
