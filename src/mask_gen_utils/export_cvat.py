import os
import cv2
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
import fiftyone as fo
from PIL import Image
from omegaconf import DictConfig
from src.utils.utils import read_yaml

# CVAT utils (import_annotations is what you used already)
import fiftyone.utils.cvat as fouc

log = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------
def _load_mask_array_or_empty(ref_image: Path, mask_path: Optional[Path]) -> np.ndarray:
    """Loads a mask if it exists; otherwise return an empty (all zeros) mask with ref image dims."""
    if mask_path and mask_path.exists():
        return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    arr = np.array(Image.open(ref_image).convert("L"), dtype=np.uint8)
    h, w = arr.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def _same_masks(mask1: Optional[np.ndarray], mask2: Optional[np.ndarray]) -> bool:
    """True iff masks are pixel-identical and both non-None."""
    if mask1 is None or mask2 is None:
        return False
    return np.array_equal(mask1, mask2)


def _combine_detection_masks_to_full_mask(
    sample: fo.Sample,
    detections_field: str = "detections",
    mask_shape_field: str = "refined_mask",
    scale_mask: int = 255,
) -> np.ndarray:
    """
    Combine instance masks from detections into a full-resolution mask aligned to mask_shape_field.
    If detections or mask shape are missing, return the existing mask of mask_shape_field.
    """
    if detections_field not in sample or mask_shape_field not in sample:
        return sample[mask_shape_field].mask

    relabeled_dets = sample[detections_field]["detections"]
    img_h, img_w = sample[mask_shape_field].mask.shape[:2]
    full = np.zeros((img_h, img_w), dtype=np.uint8)

    for det in relabeled_dets:
        det_mask = det["mask"]  # (h, w) binary
        mh, mw = det_mask.shape
        x, y, _, _ = det["bounding_box"]  # normalized xywh
        x1 = int(round(x * img_w))
        y1 = int(round(y * img_h))
        x2 = min(x1 + mw, img_w)
        y2 = min(y1 + mh, img_h)
        clip_w = x2 - x1
        clip_h = y2 - y1
        if clip_w > 0 and clip_h > 0:
            full[y1:y2, x1:x2] = det_mask[:clip_h, :clip_w] * scale_mask

    return full


# ---------------------------
# Processor
# ---------------------------
class CVATRelabelProcessor:
    """
    End-to-end helper to:
      1) Load a temp CSV
      2) Build a FiftyOne dataset from rows whose mask_status == 'cvat_uploaded'
      3) Import CVAT annotations into the dataset
      4) Materialize full relabeled masks, compare with refined, save new PNGs if changed
      5) Update the temp CSV with relabeled paths/tags/status/reviewer/timestamp and save
    """

    # Column names used in the CSV
    COL_TEMP_INITIAL_IMG = "local_developed_image_path"
    COL_TEMP_INITIAL_MASK_PATH = "temp_initial_mask_path"
    COL_TEMP_INITIAL_CUTOUT_IMG = "temp_initial_cutout_path"
    COL_TEMP_INITIAL_CUTOUT_MASK = "temp_initial_cutout_mask_path"
    COL_TEMP_REFINED_CUTOUT_MASK = "temp_refined_cutout_mask_path"
    COL_TEMP_RELABELED_CUTOUT_MASK = "temp_relabeled_cutout_mask_path"
    COL_TEMP_RELABELED_FULL_MASK = "temp_relabeled_full_mask_path"
    COL_TEMP_RELABELED_CUTOUT_IMAGE = "temp_relabeled_cutout_image_path"

    COL_STATUS = "mask_status"
    COL_REVIEWER = "mask_reviewer"
    COL_TIMESTAMP = "mask_timestamp"
    COL_TAGS = "tags"

    COL_INIT_ISSUE_TAG = "initial_mask_issue_tag"
    COL_FINAL_ISSUE_TAG = "final_mask_issue_tag"
    COL_REFINE_PARAMS = "refine_params"

    STATUS_IGNORE = ["inspected", "good", "finalized"]
    STATUS_RELABELLED = "relabelled"

    BBOX_XYWH = "bbox_xywh"  # CVAT bbox format: [x, y, width, height]

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.repo_root = Path(cfg.paths.base_dir).resolve()
        self.temp_csv_path = Path(cfg.paths.project_temp_db).resolve()
        self.output_relabeled_dir = Path(cfg.paths.relabeled_masks_dir).resolve()

        # Review/meta
        self.reviewer = os.getenv("USER") or "unknown"
        self.run_timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # CVAT
        keys = read_yaml(cfg.paths.keys_path)
        self.cvat_username = keys["cvat"]["username"]
        self.cvat_password = keys["cvat"]["password"]
        self.task_ids = list(cfg.mask_gen.export_cvat.task_ids or [])

        # Dataset/session names
        self.task_name = cfg.mask_gen.relabel.task_name or "mask_relabeling"
        self.dataset_name = f"cvat_upload_{self.task_name}"

        # Working state
        self.df: Optional[pd.DataFrame] = None
        self.dataset: Optional[fo.Dataset] = None

        # Prepare output directories
        self.output_full_masks_dir = Path(cfg.paths.project_maskgen_dir) / "final_fullsized_masks"
        self.output_full_masks_dir.mkdir(parents=True, exist_ok=True)

        self.output_cutout_images_dir = Path(cfg.paths.project_maskgen_dir) / "relabeled_cutouts"
        self.output_cutout_images_dir.mkdir(parents=True, exist_ok=True)

    # ---------- IO ----------
    def load_df(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.temp_csv_path)
        log.info(f"Loaded temp DB: {self.temp_csv_path} with {len(self.df)} rows")
        print(self.df)
        

    def save_df(self) -> None:
        assert self.df is not None, "No DataFrame loaded to save."
        self.df.to_csv(self.temp_csv_path, index=False)
        log.info(f"Saved updated temp DB: {self.temp_csv_path}")

    # ---------- Row/path helpers ----------
    def _paths_from_row(self, row: pd.Series) -> Dict[str, Optional[Path]]:
        def _resolve(p: Optional[str]) -> Optional[Path]:
            if pd.isna(p) or not p:
                return None
            return (self.repo_root / str(p)).resolve()

        return {
            "img": _resolve(row.get(self.COL_TEMP_INITIAL_CUTOUT_IMG)),
            "init_mask": _resolve(row.get(self.COL_TEMP_INITIAL_CUTOUT_MASK)),
            "refined_mask": _resolve(row.get(self.COL_TEMP_REFINED_CUTOUT_MASK)),
        }
    
    # ---------- Build samples ----------
    def build_samples(self) -> List[fo.Sample]:
        assert self.df is not None, "Call load_df() first."

        samples: List[fo.Sample] = []
        for idx, row in self.df.iterrows():
            status = str(row.get(self.COL_STATUS) or "").strip().lower()
            if status in self.STATUS_IGNORE:
                continue

            paths = self._paths_from_row(row)
            img_p = paths["img"]
            init_mask_p = paths["init_mask"]
            refined_mask_p = paths["refined_mask"]

            if not img_p or not img_p.exists():
                log.warning(f"[row {idx}] Missing image path; skipping")
                continue
            if not init_mask_p or not init_mask_p.exists():
                log.warning(f"[row {idx}] Missing initial mask path; skipping")
                continue

            s = fo.Sample(filepath=str(img_p))
            s["initial_tag"] = (row.get(self.COL_INIT_ISSUE_TAG) or "")
            s["final_tag"] = (row.get(self.COL_FINAL_ISSUE_TAG) or "")
            s["tags"] = [t.strip().lower() for t in str(row.get(self.COL_TAGS) or "").split(",") if t.strip()]
            s["status"] = status
            s["reviewer"] = (row.get(self.COL_REVIEWER) or self.reviewer)
            s["timestamp"] = (row.get(self.COL_TIMESTAMP) or self.run_timestamp)

            rp = row.get(self.COL_REFINE_PARAMS)
            s["refine_params"] = rp if (rp and isinstance(rp, str)) else "{}"

            # Attach masks
            init_arr = _load_mask_array_or_empty(img_p, init_mask_p)
            s["initial_mask"] = fo.Segmentation(mask=init_arr)

            if refined_mask_p and Path(refined_mask_p).exists():
                ref_arr = _load_mask_array_or_empty(img_p, refined_mask_p)
            else:
                ref_arr = init_arr
            s["refined_mask"] = fo.Segmentation(mask=ref_arr)

            # Placeholder that CVAT will update back into 'detections'
            s["relabeled_mask"] = fo.Segmentation(mask=ref_arr.copy())
            samples.append(s)

        log.info(f"Prepared {len(samples)} samples for CVAT import")
        return samples

    # ---------- CVAT import ----------
    def import_cvat_annotations(self, dataset_view: fo.ViewExpression) -> None:
        """
        Pulls annotations from CVAT into the dataset view. Uses task_ids from config.
        """
        if not self.task_ids:
            raise AssertionError("cfg.mask_gen.export_cvat.task_ids must be set")

        fouc.import_annotations(
            dataset_view,
            download_media=False,
            task_ids=self.task_ids,
            username=self.cvat_username,
            password=self.cvat_password,
        )
        log.info(f"Imported CVAT annotations for tasks: {self.task_ids}")

    # ---------- Mask materialization & save ----------
    def _save_relabeled_mask_png(self, sample: fo.Sample) -> Path:
        """Save relabeled mask next to configured output dir with _mask.png suffix; returns path."""
        self.output_relabeled_dir.mkdir(parents=True, exist_ok=True)
        image_name = Path(sample.filepath).name
        out_path = (self.output_relabeled_dir / image_name).with_suffix("").with_name(
            Path(image_name).stem + "_mask.png"
        )
        cv2.imwrite(str(out_path), sample["relabeled_mask"].mask)
        log.info(f"Saved relabeled mask: {out_path}")
        return out_path

    def realize_and_compare_masks(self, dataset_view: fo.ViewExpression) -> Dict[str, Dict[str, str]]:
        """
        Builds full masks from detections, compares to refined masks, and saves PNGs if updated.

        Returns a dict keyed by image absolute path with per-sample updates:
          {
            "/abs/path/to/image.jpg": {
              "status": "relabelled" | original_status,
              "mask_path": "/abs/path/to/saved/_mask.png" | "",
              "reviewer": "...",
              "timestamp": "..."
            },
            ...
          }
        """
        updates: Dict[str, Dict[str, str]] = {}

        for sample in dataset_view:
            # 1. Build full relabeled mask (or keep refined if no detections)
            full_mask = _combine_detection_masks_to_full_mask(sample)
            sample["relabeled_mask"] = fo.Segmentation(mask=full_mask)

            # Clean up detections (keep dataset lean)
            if "detections" in sample:
                del sample["detections"]

            refined_mask = sample["refined_mask"].mask if "refined_mask" in sample else None
            relabeled_mask = sample["relabeled_mask"].mask

            changed = not _same_masks(refined_mask, relabeled_mask)
            if changed:
                status = self.STATUS_RELABELLED
                reviewer = self.reviewer
                timestamp = self.run_timestamp
                mask_path = str(self._save_relabeled_mask_png(sample))
            else:
                # No change; keep original meta
                status = str(sample["status"] or "").strip().lower()
                reviewer = sample["reviewer"] or self.reviewer
                timestamp = sample["timestamp"] or self.run_timestamp
                mask_path = ""  # don't create new file

            updates[str(Path(sample.filepath).resolve())] = {
                "status": status,
                "reviewer": reviewer,
                "timestamp": timestamp,
                "mask_path": mask_path,
                # You can also choose to propagate 'tags' or 'final_tag' changes here if desired
            }

        return updates
    
    def _resize_mask_nearest(self, mask: np.ndarray, w: int, h: int) -> np.ndarray:
        if mask.shape[1] == w and mask.shape[0] == h:
            return mask
        log.warning(f"Resizing mask from {mask.shape[1]}x{mask.shape[0]} to {w}x{h}")
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---------- CSV updates ----------
    def update_dataframe_with_results(self, sample_updates: Dict[str, Dict[str, str]]) -> None:
        """
        Applies updates back to self.df and writes relabeled mask paths/status/etc.
        We match rows by the absolute resolved path of COL_TEMP_INITIAL_CUTOUT_IMG.
        """
        assert self.df is not None, "Call load_df() first."

        # Normalize a mapping from abs image path -> row indices
        path_to_idx: Dict[str, List[int]] = {}
        for idx, row in self.df.iterrows():
            p = row.get(self.COL_TEMP_INITIAL_CUTOUT_IMG)
            if pd.isna(p) or not p:
                continue
            abs_img = str((self.repo_root / str(p)).resolve())
            path_to_idx.setdefault(abs_img, []).append(idx)

        apply_count = 0

        for abs_img_path, meta in sample_updates.items():
            if abs_img_path not in path_to_idx:
                continue
            for idx in path_to_idx[abs_img_path]:
                # Update status, reviewer, timestamp
                self.df.at[idx, self.COL_STATUS] = meta["status"]
                self.df.at[idx, self.COL_REVIEWER] = meta["reviewer"]
                self.df.at[idx, self.COL_TIMESTAMP] = meta["timestamp"]

                # If we saved a new mask, make a repo-relative path for consistency
                if meta["mask_path"]:
                    abs_mask = Path(meta["mask_path"]).resolve()
                    try:
                        rel_to_repo = abs_mask.relative_to(self.repo_root)
                        self.df.at[idx, self.COL_TEMP_RELABELED_CUTOUT_MASK] = str(rel_to_repo)
                    except ValueError:
                        # Different root; store absolute path
                        self.df.at[idx, self.COL_TEMP_RELABELED_CUTOUT_MASK] = str(abs_mask)
                apply_count += 1

        log.info(f"Applied updates to {apply_count} row(s) in temp DB")

    def create_fullsize_masks_from_cutouts(self) -> None:
        """
        For each row with a relabeled cutout mask and bbox_xywh, create a full-sized mask
        whose shape matches `temp_initial_mask_path` and paste the cutout mask into it at bbox.

        Writes output PNG and updates `temp_relabeled_full_mask_path`.
        """
        assert self.df is not None, "Call load_df() first."


        updated = 0

        for idx, row in self.df.iterrows():
            rel_cutout_str = row.get(self.COL_TEMP_RELABELED_CUTOUT_MASK, "")
            bbox_val       = row.get(self.BBOX_XYWH, None)
            full_ref_str   = row.get(self.COL_TEMP_INITIAL_MASK_PATH, "")

            if not rel_cutout_str or pd.isna(rel_cutout_str) or not bbox_val or pd.isna(bbox_val) or not full_ref_str:
                continue

            rel_cutout_path = (self.repo_root / str(rel_cutout_str)).resolve()
            full_ref_path   = (self.repo_root / str(full_ref_str)).resolve()

            if not rel_cutout_path.exists():
                log.warning(f"[row {idx}] Missing relabeled cutout mask: {rel_cutout_path}")
                continue
            if not full_ref_path.exists():
                log.warning(f"[row {idx}] Missing full-size reference (temp_initial_mask_path): {full_ref_path}")
                continue

            # Parse bbox_xywh (accept JSON "[x,y,w,h]" or "x,y,w,h" or list)
            try:
                if isinstance(bbox_val, str):
                    bbox = json.loads(bbox_val) if bbox_val.strip().startswith("[") else [float(v) for v in bbox_val.split(",")]
                elif isinstance(bbox_val, (list, tuple, np.ndarray, pd.Series)):
                    bbox = list(bbox_val)
                else:
                    raise ValueError("Unsupported bbox type")
                x, y, bw, bh = [int(round(float(v))) for v in bbox[:4]]
            except Exception as e:
                log.warning(f"[row {idx}] Could not parse bbox_xywh='{bbox_val}': {e}")
                continue

            # Load canvas (get H,W from full_ref_path)
            try:
                # open as grayscale just to get shape; we don't care about values
                full_ref = np.array(Image.open(full_ref_path).convert("L"), dtype=np.uint8)
            except Exception as e:
                log.warning(f"[row {idx}] Failed reading full ref size from {full_ref_path}: {e}")
                continue

            H, W = full_ref.shape[:2]
            canvas = np.zeros((H, W), dtype=np.uint8)

            # Load the relabeled cutout mask as uint8 {0,255}
            try:
                cutout = np.array(Image.open(rel_cutout_path).convert("L"), dtype=np.uint8)
            except Exception as e:
                log.warning(f"[row {idx}] Failed reading relabeled cutout mask {rel_cutout_path}: {e}")
                continue

            # Clip bbox to canvas
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(W, x + bw)
            y2 = min(H, y + bh)
            if x1 >= x2 or y1 >= y2:
                log.warning(f"[row {idx}] BBox out of bounds after clipping: {(x,y,bw,bh)} on {(W,H)}")
                continue

            # Ensure cutout matches clipped bbox size
            clip_w = x2 - x1
            clip_h = y2 - y1
            cutout_resized = self._resize_mask_nearest(cutout, clip_w, clip_h)

            # Paste (OR/union semantics; if you prefer overwrite, just assign)
            region = canvas[y1:y2, x1:x2]
            np.maximum(region, cutout_resized, out=region)

            # Save
            base_name = Path(row.get(self.COL_TEMP_INITIAL_IMG, Path(rel_cutout_path).stem)).stem
            out_path = self.output_full_masks_dir / f"{base_name}_mask.png"
            cv2.imwrite(str(out_path), canvas)

            # Update CSV with repo-relative path if possible
            try:
                rel_to_repo = out_path.resolve().relative_to(self.repo_root)
                self.df.at[idx, self.COL_TEMP_RELABELED_FULL_MASK] = str(rel_to_repo)
            except Exception:
                self.df.at[idx, self.COL_TEMP_RELABELED_FULL_MASK] = str(out_path.resolve())

            updated += 1

        log.info(f"Created {updated} full-sized mask(s) from relabeled cutouts.")
    
    def create_masked_cutout_images(self) -> None:
        """
        Create a new masked cutout image using the relabeled cutout mask.
        Saves as RGBA PNG with transparency from the mask.
        """
        assert self.df is not None, "Call load_df() first."

        made = 0
        for idx, row in self.df.iterrows():
            cutout_img_str  = row.get(self.COL_TEMP_INITIAL_CUTOUT_IMG, "")
            rel_mask_str    = row.get(self.COL_TEMP_RELABELED_CUTOUT_MASK, "")

            if not cutout_img_str or pd.isna(cutout_img_str):
                continue
            if not rel_mask_str or pd.isna(rel_mask_str):
                continue

            cutout_img_path = (self.repo_root / str(cutout_img_str)).resolve()
            rel_mask_path   = (self.repo_root / str(rel_mask_str)).resolve()

            if not cutout_img_path.exists():
                log.warning(f"[row {idx}] Missing cutout image: {cutout_img_path}")
                continue
            if not rel_mask_path.exists():
                log.warning(f"[row {idx}] Missing relabeled cutout mask: {rel_mask_path}")
                continue

            # Load cutout image (BGR)
            cutout_bgr = cv2.imread(str(cutout_img_path), cv2.IMREAD_COLOR)
            if cutout_bgr is None:
                log.warning(f"[row {idx}] Failed to read cutout image: {cutout_img_path}")
                continue
            H, W = cutout_bgr.shape[:2]

            # Load mask (grayscale), resize if needed
            mask_gray = cv2.imread(str(rel_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                log.warning(f"[row {idx}] Failed to read relabeled mask: {rel_mask_path}")
                continue
            if mask_gray.shape[:2] != (H, W):
                mask_gray = self._resize_mask_nearest(mask_gray, W, H)

            # Robust binarization: treat >0 as foreground
            # (works for both 0/1 and 0/255 masks)
            mask_bin = (mask_gray > 0).astype(np.uint8)

            # Expand to 3 channels and apply
            mask_3c = np.repeat(mask_bin[:, :, None], 3, axis=2)
            masked_bgr = cutout_bgr * mask_3c  # background → black

            # Save PNG with transparency
            base_stem = Path(cutout_img_path).stem
            out_path = self.output_cutout_images_dir / f"{base_stem}.png"
            if not cv2.imwrite(str(out_path), masked_bgr):
                log.warning(f"[row {idx}] Failed to write masked cutout image: {out_path}")
                continue

            # Update CSV path
            try:
                rel_to_repo = out_path.resolve().relative_to(self.repo_root)
                self.df.at[idx, self.COL_TEMP_RELABELED_CUTOUT_IMAGE] = str(rel_to_repo)
            except Exception:
                self.df.at[idx, self.COL_TEMP_RELABELED_CUTOUT_IMAGE] = str(out_path.resolve())

            made += 1

        log.info(f"Created {made} masked cutout image(s) with transparency.")
    # ---------- Orchestration ----------
    def run(self) -> None:
        # 1) Load CSV
        df = self.load_df()

        # 2) Build samples
        samples = self.build_samples()
        if not samples:
            log.info("No eligible samples (status == 'cvat_uploaded'); nothing to do.")
            return

        # 3) Make dataset
        if fo.dataset_exists(self.dataset_name):
            fo.delete_dataset(self.dataset_name)
        self.dataset = fo.Dataset(self.dataset_name)
        self.dataset.add_samples(samples)
        view = self.dataset.view()  # whole dataset

        # 4) Import from CVAT
        self.import_cvat_annotations(view)

        # 5) Realize masks & collect per-sample changes
        sample_updates = self.realize_and_compare_masks(view)

        # 6) Updates back to CSV
        self.update_dataframe_with_results(sample_updates)

        # 7) Create full-sized masks from cutouts
        self.create_fullsize_masks_from_cutouts()

        # 8) Create masked cutout images
        self.create_masked_cutout_images()

        # 9) Save the updated DataFrame back to CSV
        self.save_df()


# ---------------------------
# Hydra Entry
# ---------------------------
@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Config expectations (keys used here):
      cfg.paths.base_dir
      cfg.paths.project_temp_db
      cfg.paths.relabeled_masks_dir
      cfg.paths.keys_path
      cfg.mask_gen.export_cvat.task_ids  (list of CVAT task IDs)
      cfg.mask_gen.relabel.task_name
      cfg.mask_gen.relabel.cvat_url (optional; defaults to 'http://sunny.ece.ncsu.edu:8080/')
    """
    # TODO: Create new cutout from relabeled cutout mask
    processor = CVATRelabelProcessor(cfg)
    processor.run()


if __name__ == "__main__":
    main()
