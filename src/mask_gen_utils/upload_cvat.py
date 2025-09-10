import os
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import fiftyone as fo
import numpy as np
import pandas as pd
from PIL import Image

import hydra
from omegaconf import DictConfig

from src.utils.utils import read_yaml

log = logging.getLogger(__name__)

# Columns we expect / write back into the temp CSV
NEEDED_COLS = [
    "temp_initial_cutout_path",                        # path to cutout image (jpg/png)
    "temp_initial_cutout_mask_path",        # path to initial cutout mask
    "temp_refined_cutout_mask_path",        # path to refined mask (may be empty)
    "temp_relabeled_cutout_mask_path",      # path to relabeled mask (we'll fill later)
    "initial_mask_issue_tag",          # e.g., "missing_red"
    "final_mask_issue_tag",            # optional override/after-inspect tag
    "tags",                            # comma separated
    "mask_status",                     # unreviewed|refined|cvat_uploaded|relabeled|...
    "mask_reviewer",
    "mask_timestamp",
    "refine_params",                   # json string
]

class UploadToCVAT:
    """
    Upload a subset of samples to CVAT for relabeling, using the shared temp CSV as the DB.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.repo_root = Path(cfg.paths.base_dir).resolve()
        self.maskgen_dir = Path(cfg.paths.project_maskgen_dir).resolve()
        self.cutout_dir = self.maskgen_dir / "cutouts"
        self.refined_mask_dir = self.maskgen_dir / "refined_masks"

        self.temp_csv = Path(cfg.paths.project_temp_db)
        if not self.temp_csv.exists():
            raise FileNotFoundError(f"project_temp_db CSV not found: {self.temp_csv}")

        self.keys = read_yaml(cfg.paths.keys_path)
        self.annot_session_key = cfg.mask_gen.relabel.annot_session_key
        self.task_name = cfg.mask_gen.relabel.task_name or "mask_relabeling"
        self.dest_field = "relabeled_mask"

        self.only_tags = list(cfg.mask_gen.relabel.only_tags or [])
        self.reviewer = os.getenv("USER") or "unknown"
        self.timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        self.df = pd.read_csv(self.temp_csv)
        self._ensure_columns()

    # ---------- CSV helpers ----------

    def _ensure_columns(self) -> None:
        for col in NEEDED_COLS:
            if col not in self.df.columns:
                self.df[col] = pd.Series([None] * len(self.df), dtype="object")
            else:
                self.df[col] = self.df[col].astype("object")

    def _save_csv(self) -> None:
        self.df.to_csv(self.temp_csv, index=False)
        log.info(f"Wrote updates to temp CSV: {self.temp_csv}")

    # ---------- Row selection ----------

    def _split_tags(self, val) -> List[str]:
        if pd.isna(val) or val is None:
            return []
        return [t.strip().lower() for t in str(val).split(",") if t.strip()]

    def _row_has_target_tag(self, row: pd.Series) -> Optional[str]:
        """
        Return the first tag that is in our processor namespace if it matches only_tags (when provided).
        We check final_mask_issue_tag, then initial_mask_issue_tag, then tags.
        """
        candidates: List[str] = []
        candidates += self._split_tags(row.get("final_mask_issue_tag"))
        if not candidates:
            candidates += self._split_tags(row.get("initial_mask_issue_tag"))
        if not candidates:
            candidates += self._split_tags(row.get("tags"))

        for t in candidates:
            if not self.only_tags or t in self.only_tags:
                return t
        return None

    def _resolve_paths(self, row: pd.Series) -> Optional[Dict[str, Path]]:
        # image path

        fp = row.get("temp_initial_cutout_path")
        if not fp:
            log.warning("Row has no initial cutout path; skipping")
            return None
        img_path = Path(str(fp))
        if not img_path.is_absolute():
            img_path = (self.repo_root / img_path).resolve()
        if not img_path.exists():
            log.warning(f"Row has missing image path {img_path}; skipping")
            return None

        # initial mask (prefer explicit column)
        m0 = row.get("temp_initial_cutout_mask_path")
        if pd.notna(m0) and m0:
            m0p = Path(str(m0))
            if not m0p.is_absolute():
                m0p = (self.repo_root / m0p).resolve()
        else:
            cname = Path(img_path).stem + "_0"
            m0p = (self.cutout_dir / str(cname).replace(".jpg", "_mask.png")).resolve()

        if not m0p.exists():
            # no initial: we can still upload with an empty mask
            m0p = None

        # refined mask
        mr = row.get("temp_refined_cutout_mask_path")
        if pd.notna(mr) and mr:
            mrp = Path(str(mr))
            if not mrp.is_absolute():
                mrp = (self.repo_root / mrp).resolve()
            if not mrp.exists():
                mrp = None
        else:
            mrp = None

        return {"image": img_path, "mask_initial": m0p, "mask_refined": mrp}

    # ---------- FiftyOne dataset building ----------

    def _load_mask_array_or_empty(self, ref_image: Path, mask_path: Optional[Path]) -> np.ndarray:
        if mask_path and mask_path.exists():
            return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        # empty with image dims
        arr = np.array(Image.open(ref_image).convert("L"), dtype=np.uint8)
        h, w = arr.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    def _gather_samples(self) -> Tuple[List[fo.Sample], List[int]]:
        rows_idx: List[int] = []
        samples: List[fo.Sample] = []

        for idx, row in self.df.iterrows():
            # Skip rows without detections results
            if not pd.isna(row.get("detection_note")) and row.get("detection_note").lower() == "no detection":
                log.warning(f"Row {idx} has no detections; skipping")
                continue

            # Skip rows already uploaded or finished unless you want to re-upload
            status = (row.get("mask_status") or "").strip().lower()
            if status in {"cvat_uploaded", "relabeled"}:
                log.warning(f"Skipping row {idx} with status '{status}'")
                continue

            t = self._row_has_target_tag(row)
            if t is None:
                log.warning(f"Row {idx} has no target tag; skipping")
                continue

            paths = self._resolve_paths(row)
            if not paths or not paths["image"].exists():
                log.warning(f"Row {idx} has missing image or mask paths; skipping")
                continue

            # Build sample
            s = fo.Sample(filepath=str(paths["image"]))
            s["initial_tag"] = str(row.get("initial_mask_issue_tag") or "")  # keep raw
            s["final_tag"] = str(row.get("final_mask_issue_tag") or "")
            s["tags"] = self._split_tags(row.get("tags"))

            # carry reviewer/timestamp/refine_params forward
            s["status"] = status or ""
            s["reviewer"] = str(row.get("mask_reviewer") or self.reviewer)
            s["timestamp"] = str(row.get("mask_timestamp") or self.timestamp)
            rp = row.get("refine_params")
            s["refine_params"] = rp if (rp and isinstance(rp, str)) else "{}"

            # attach masks
            init_arr = self._load_mask_array_or_empty(paths["image"], paths["mask_initial"])
            s["initial_mask"] = fo.Segmentation(mask=init_arr)

            if paths["mask_refined"] and paths["mask_refined"].exists():
                ref_arr = self._load_mask_array_or_empty(paths["image"], paths["mask_refined"])
            else:
                ref_arr = init_arr.copy()
            s["refined_mask"] = fo.Segmentation(mask=ref_arr)

            # placeholder relabeled (CVAT will write back)
            s["relabeled_mask"] = fo.Segmentation(mask=ref_arr.copy())

            samples.append(s)
            rows_idx.append(idx)

        return samples, rows_idx

    # ---------- CVAT ----------

    def _send_to_cvat(self, view: fo.DatasetView) -> None:
        schema = {
            self.dest_field: {
                "label_type": "Segmentation",
                "classes": ["weed"],
                "mask_targets": {255: "weed"},
            }
        }
        view.annotate(
            self.annot_session_key,
            backend="cvat",
            label_schema=schema,
            launch_editor=False,
            username=self.keys["cvat"]["username"],
            password=self.keys["cvat"]["password"],
            task_name=self.task_name,
        )
        log.info("CVAT task created. Open CVAT UI to annotate.")

    # ---------- Public run ----------

    def run(self) -> None:
        samples, row_ids = self._gather_samples()
        if not samples:
            log.info("No eligible samples found to upload to CVAT.")
            return

        # make dataset
        dataset_name = f"cvat_upload_{self.task_name}_{self.timestamp.replace(':','-')}"
        ds = fo.Dataset(dataset_name)
        ds.add_samples(samples)
        ds.save()

        view = ds.view()  # could filter/sort further
        self._send_to_cvat(view)

        # mark uploaded in CSV
        for i in row_ids:
            self.df.at[i, "mask_status"] = "cvat_uploaded"
            # keep reviewer/timestamp coherent
            self.df.at[i, "mask_reviewer"] = self.reviewer
            self.df.at[i, "mask_timestamp"] = self.timestamp
            # ensure refine_params is at least "{}"
            if not self.df.at[i, "refine_params"]:
                self.df.at[i, "refine_params"] = "{}"

        self._save_csv()
        log.info(f"Uploaded {len(row_ids)} samples to CVAT and updated CSV.")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Starting CVAT upload (temp_db mode)")
    try:
        UploadToCVAT(cfg).run()
    except Exception as e:
        log.error(f"Fatal error in upload_cvat: {e}", exc_info=True)
    log.info("Finished CVAT upload")

if __name__ == "__main__":
    main()
