import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.mask_gen_utils.missing_red import MissingRed
from src.mask_gen_utils.missing_white import MissingWhite
from src.mask_gen_utils.present_mat import PresentMat

log = logging.getLogger(__name__)

# Columns we’ll ensure exist in the temp CSV
REFINE_COLS = [
    "temp_refined_cutout_mask_path",
    "refine_params",              # JSON string of the config used for this refine op
    "mask_status",                # keep in sync with inspect.py
    "mask_reviewer",
    "mask_review_datetime",
]

class RefineMask:
    """
    Refine cutout masks using rule-based processors.
    Reads rows from the shared temp CSV, writes refined mask paths + metadata back.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Paths
        self.repo_root = Path(cfg.paths.base_dir).resolve()
        self.maskgen_dir = Path(cfg.paths.project_maskgen_dir).resolve()
        self.cutout_dir = self.maskgen_dir / "cutouts"
        self.refined_mask_dir = self.maskgen_dir / "refined_masks"
        self.refined_mask_dir.mkdir(parents=True, exist_ok=True)

        # Input/Output “temp db”
        self.temp_csv = Path(cfg.paths.project_temp_db)

        # Which tags to process (optional)
        self.only_tags: Optional[List[str]] = list(cfg.mask_gen.refine.only_tags or [])

        # Reviewer + timestamp
        self.reviewer = os.getenv("USER") or "unknown"
        self.run_timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # Processor registry and their cfg (for persistable refine_params)
        refine_cfg = cfg.mask_gen.refine
        self.processors: Dict[str, Tuple[Any, Any]] = {
            "missing_red":   (MissingRed(refine_cfg.missing_red), refine_cfg.missing_red),
            "missing_white": (MissingWhite(refine_cfg.missing_white), refine_cfg.missing_white),
            "present_mat":   (PresentMat(refine_cfg.present_mat), refine_cfg.present_mat),
        }

        # Load CSV
        if not self.temp_csv.exists():
            raise FileNotFoundError(f"project_temp_db CSV not found: {self.temp_csv}")
        self.df = pd.read_csv(self.temp_csv)
        self._ensure_refine_columns()

    # ---------------- CSV helpers ----------------

    def _ensure_refine_columns(self) -> None:
        for col in REFINE_COLS:
            if col not in self.df.columns:
                self.df[col] = pd.Series([None] * len(self.df), dtype="object")
            else:
                self.df[col] = self.df[col].astype("object")

    def _save_csv(self) -> None:
        self.df.to_csv(self.temp_csv, index=False)
        log.info(f"Wrote updates to temp CSV: {self.temp_csv}")

    # ---------------- Row filtering ----------------

    def _row_needs_refine(self, row: pd.Series) -> Tuple[Optional[str], bool]:
        """
        Decide whether this row should be refined and return the selected tag key.

        We check, in order:
          - final_mask_issue_tag
          - initial_mask_issue_tag
          - tags (comma-separated)
        We match the first tag that has a registered processor key.

        If only_tags is provided, we require the chosen tag to be in only_tags.
        """
        candidates: List[str] = []

        def split_tags(val) -> List[str]:
            if pd.isna(val) or val is None:
                return []
            return [t.strip().lower() for t in str(val).split(",") if t.strip()]

        candidates += split_tags(row.get("final_mask_issue_tag"))
        if not candidates:
            candidates += split_tags(row.get("initial_mask_issue_tag"))
        if not candidates:
            candidates += split_tags(row.get("tags"))

        # pick the first tag that maps to one of our processors
        for t in candidates:
            if t in self.processors:
                # respect only_tags if provided
                if self.only_tags and (t not in self.only_tags):
                    return None, False
                return t, True

        return None, False

    # ---------------- Path resolution ----------------

    def _resolve_paths(self, row: pd.Series) -> Optional[Dict[str, Path]]:
        """
        Resolve the existing cutout image and its (initial) mask path.
        Prefers 'initial_cutout_mask_path' as written by segment.py.
        """
        mask_path = None
        val = row.get("temp_initial_cutout_mask_path", None)
        if pd.notna(val) and val:
            p = Path(str(val))
            if not p.is_absolute():
                p = (self.repo_root / p).resolve()
            mask_path = p if p.exists() else None

        if mask_path is None:
            # Try to infer from cutout_name if present
            cname = row.get("cutout_name", None)
            if pd.notna(cname) and cname:
                candidate = (self.cutout_dir / str(cname).replace(".jpg", "_mask.png")).resolve()
                mask_path = candidate if candidate.exists() else None

        if mask_path is None or not mask_path.exists():
            return None

        # Derive image path (prefer .jpg, then .png)
        img_jpg = Path(str(mask_path).replace("_mask.png", ".jpg"))
        img_png = Path(str(mask_path).replace("_mask.png", ".png"))
        image_path = img_jpg if img_jpg.exists() else (img_png if img_png.exists() else None)
        if image_path is None or not image_path.exists():
            return None

        return {"image": image_path, "mask": mask_path}

    # ---------------- Core processing ----------------

    def _run_processor(self, tag_key: str, image_bgr: np.ndarray, mask_gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        proc, proc_cfg = self.processors[tag_key]
        # Convert BGR->RGB if your processors expect RGB; keep as-is otherwise.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        refined = proc.process(image_rgb, mask_gray)  # Expecting np.uint8 mask (0/255 or 0/1)
        params = OmegaConf.to_container(proc_cfg, resolve=True)
        return refined, params

    def _save_refined(self, refined_mask: np.ndarray, src_mask_path: Path) -> Path:
        """
        Save refined mask alongside project refined_masks dir.
        We mirror the cutout mask's filename into refined_masks/.
        """
        name = src_mask_path.name  # e.g., <stem>_0_mask.png
        out_path = (self.refined_mask_dir / name).resolve()

        # Normalize to 0/255 uint8 before saving
        m = refined_mask
        if m.dtype != np.uint8:
            m = (m.astype(np.float32) > 0.5).astype(np.uint8) * 255
        elif m.max() == 1:  # binary {0,1}
            m = (m * 255).astype(np.uint8)

        cv2.imwrite(str(out_path), m)
        return out_path

    # ---------------- Public run ----------------

    def run(self) -> None:
        updated, skipped, missing = 0, 0, 0

        for idx, row in self.df.iterrows():
            tag_key, should = self._row_needs_refine(row)
            if not should or tag_key is None:
                skipped += 1
                continue

            paths = self._resolve_paths(row)
            if not paths:
                log.warning(f"Row {idx} ({tag_key}) has no valid image/mask paths")
                missing += 1
                continue

            image_bgr = cv2.imread(str(paths["image"]), cv2.IMREAD_COLOR)
            mask_gray = cv2.imread(str(paths["mask"]), cv2.IMREAD_GRAYSCALE)
            if image_bgr is None or mask_gray is None:
                missing += 1
                continue

            try:
                refined_mask, params = self._run_processor(tag_key, image_bgr, mask_gray)
                out_path = self._save_refined(refined_mask, paths["mask"])

                # Prefer repo-relative path when possible
                try:
                    rel = out_path.relative_to(self.repo_root)
                except ValueError:
                    rel = out_path

                # Update row
                self.df.at[idx, "temp_refined_cutout_mask_path"] = str(rel)
                self.df.at[idx, "refine_params"] = json.dumps(params)
                self.df.at[idx, "mask_status"] = "refined"
                self.df.at[idx, "mask_reviewer"] = self.reviewer
                self.df.at[idx, "mask_review_datetime"] = self.run_timestamp
                updated += 1

            except Exception as e:
                log.warning(f"Refine error on row {idx} ({tag_key}): {e}")
                skipped += 1

        log.info(f"Refine complete — updated: {updated}, skipped: {skipped}, missing: {missing}")
        self._save_csv()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Starting mask refine (temp_db mode)")
    try:
        RefineMask(cfg).run()
    except Exception as e:
        log.error(f"Fatal error in refine: {e}", exc_info=True)
    log.info("Finished mask refine")

if __name__ == "__main__":
    main()
