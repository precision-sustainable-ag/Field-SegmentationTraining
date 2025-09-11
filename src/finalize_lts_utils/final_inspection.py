#!/usr/bin/env python3
"""
Final Mask Inspection & LTS Handoff (temp_db mode)

Workflow
--------
1) Load the project temp CSV (same artifact used across the pipeline)
2) Resolve the *latest* mask path per row (relabeled > refined > initial)
3) Build a FiftyOne dataset to review/approve/reject
4) Reviewer uses tags to indicate final disposition
5) After session closes:
   - Write tags + final status back to the temp CSV
   - For "final-ok": move cutout + full-size mask to LTS and record LTS paths
   - Update persistent DB with final status/paths (if configured)

Config keys (OmegaConf)
-----------------------
paths:
  base_dir: "/abs/path/to/repo-root"
  project_maskgen_dir: "/abs/path/to/project/mask_gen"
  project_temp_db: "/abs/path/to/project/_temp_db.csv"
  field_cutouts_dir: "/abs/path/to/LTS/field-cutouts"
  field_masks_dir: "/abs/path/to/LTS/field-masks"
  agir_field_db: "/abs/path/to/persistent.db"   # optional
  table_name: "mask_gen_images"                 # optional
  primary_key_field: "cutout_id"                # optional

mask_gen:
  final_inspect:
    port: 5150
    dataset_name: "maskgen-final-inspect"
    only_tags: null
    remove_source: false
    canonical_final_tag_mapping:
      final_ok: "approve"
      final_reject: "reject"
      final_needsfix: "needsfix"
      final_defer: "defer"
"""

import os
import re
import shutil
import sqlite3
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import fiftyone as fo
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# ---------------- constants ----------------

FINAL_COLS = [
    # existing inspection fields
    "initial_mask_issue_tag", 
    "final_mask_issue_tag", 
    "tags",
    "mask_status", 
    "mask_reviewer", 
    "mask_timestamp",
    "final_cutout_path", 
    "final_mask_path",
]

# Latest CUTOUT-MASK preference (cutout-sized masks)
LATEST_CUTOUT_MASK_PRIORITY = [
    "temp_relabeled_cutout_mask_path",
    "temp_refined_cutout_mask_path",
    "temp_initial_cutout_mask_path",
]

# Latest FULL-SIZE mask preference
LATEST_FULL_MASK_PRIORITY = [
    "temp_relabeled_full_mask_path",   # created by export_cvat.py
    "temp_initial_mask_path",          # created by segment.py
]

# Primary cutout image produced by segment.py
STATIC_INITIAL_CUTOUT_PATH = "temp_initial_cutout_path"

MASK_STATUS_FOR_ELIGIBLE_REVIEW = {
    "unreviewed", 
    # "final-approved", 
    "final-reject",
    "final-needsfix", 
    "final-defer", 
    "relabeled",
    "unreviewed",
    "cvat_uploaded",
    }

# ---------------- helper class ----------------

class FinalMaskInspector:
    """
    Final review + handoff to LTS driven by the project temp CSV.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Paths
        self.base_dir = Path(cfg.paths.base_dir).resolve()
        self.project_dir = Path(cfg.paths.project_maskgen_dir)
        self.temp_csv = Path(cfg.paths.project_temp_db)

        # Common project dirs produced earlier in the pipeline
        self.cutouts_dir = self.project_dir / "cutouts"

        # LTS targets
        self.lts_cutouts_dir = Path(cfg.paths.field_cutouts_dir).resolve()
        self.lts_masks_dir = Path(cfg.paths.field_masks_dir).resolve()
        self.lts_cutouts_dir.mkdir(parents=True, exist_ok=True)
        self.lts_masks_dir.mkdir(parents=True, exist_ok=True)
        self.remove_source = bool(getattr(cfg.mask_gen.final_inspect, "remove_source", False))

        # FiftyOne
        self.port = int(cfg.mask_gen.final_inspect.port)
        self.dataset_name = cfg.mask_gen.final_inspect.dataset_name
        self.only_tags: Optional[List[str]] = cfg.mask_gen.final_inspect.only_tags

        # Canonical tag mapping for final decisions
        self.canonical_map: Dict[str, str] = dict(
            getattr(cfg.mask_gen.final_inspect, "canonical_final_tag_mapping", {})
        )

        # Persistent DB (optional)
        self.db_path = Path(getattr(cfg.paths, "agir_field_db", "")) if hasattr(cfg.paths, "agir_field_db") else None
        self.db_table = cfg.paths.table_name
        self.db_pk = cfg.paths.primary_key_field

        # Reviewer
        self.reviewer = os.getenv("USER") or "unknown"
        self.timestamp = datetime.datetime.now().isoformat()

        # Runtime state
        self.dataset: Optional[fo.Dataset] = None
        self.session: Optional[fo.Session] = None

        # CSV
        self.df = self._load_csv()
        self._ensure_final_columns()

    # ---------- CSV I/O ----------

    def _load_csv(self) -> pd.DataFrame:
        if not self.temp_csv.exists():
            raise FileNotFoundError(f"Temp CSV not found: {self.temp_csv}")
        df = pd.read_csv(self.temp_csv)
        log.info(f"Loaded temp CSV with {len(df)} rows: {self.temp_csv}")
        return df

    def _save_csv(self) -> None:
        self.df = self.df.reindex(sorted(self.df.columns), axis=1)
        self.df.to_csv(self.temp_csv, index=False)
        log.info(f"Wrote updates to temp CSV: {self.temp_csv}")

    def _ensure_final_columns(self) -> None:
        for col in FINAL_COLS:
            if col not in self.df.columns:
                self.df[col] = pd.Series([None] * len(self.df), dtype="object")
            else:
                self.df[col] = self.df[col].astype("object")

    # ---------- Path resolvers ----------

    def _pick_first_existing(self, row: pd.Series, keys: List[str]) -> Optional[Path]:
        for k in keys:
            v = row.get(k, None)
            if pd.isna(v) or v is None:
                continue
            p = Path(str(v))
            if not p.is_absolute():
                p = (self.base_dir / p).resolve()
            if p.exists():
                return p
        return None

    def _resolve_cutout_mask_path(self, row: pd.Series) -> Optional[Path]:
        return self._pick_first_existing(row, LATEST_CUTOUT_MASK_PRIORITY)

    def _resolve_fullsized_mask_path(self, row: pd.Series) -> Optional[Path]:
        """
        Derive the full-sized mask path from the cutout mask path.

        Typical naming:
        <stem>_<idx>_mask.png  ->  <stem>_mask.png

        If the cutout mask lives in a 'cutouts' dir, the full-size mask is expected
        under '<project_dir>/developed-images/'. Otherwise, assume it is a sibling.
        """
        cutout_mask_path = self._resolve_cutout_mask_path(row)
        if not cutout_mask_path:
            return None

        cutout_mask_path = cutout_mask_path.resolve()
        name = cutout_mask_path.name

        # strip a single `_digits` right before `_mask.png`
        # e.g., foo_12_mask.png -> foo_mask.png ; foo_0_mask.png -> foo_mask.png
        full_name = re.sub(r"_(\d+)(?=_mask\.png$)", "", name, flags=re.IGNORECASE)

        mask_parent = cutout_mask_path.parent
        if mask_parent.name == "cutouts":
            fullframe_mask_path = (self.project_dir / "developed-images" / full_name).resolve()
            log.info(f"Resolved full-size mask (from cutouts dir): {fullframe_mask_path}")
        else:
            fullframe_mask_path = (mask_parent / full_name).resolve()
            log.info(f"Resolved full-size mask (sibling dir): {fullframe_mask_path}")

        if fullframe_mask_path.exists():
            return fullframe_mask_path

        log.warning(f"Full-sized mask path does not exist: {fullframe_mask_path}")
        return None

    def _resolve_cutout_path(self, row: pd.Series, cropout: bool = False) -> Optional[Path]:
        """
        Derive the cutout image filepath from the cutout mask filepath.
        """
        cutout_mask_path = self._resolve_cutout_mask_path(row)
        if not cutout_mask_path:
            return None

        cutout_mask_path = cutout_mask_path.resolve()
        cdir = cutout_mask_path.parent
        # remove the '_mask' token right before extension
        base_no_mask = cutout_mask_path.name.replace("_mask", "")
        cand = cdir / base_no_mask

        if cropout:
            # replace the png with jpg using replace
            base_no_mask = row[STATIC_INITIAL_CUTOUT_PATH]
            cand = Path(base_no_mask)

        if cand.exists():
            return cand

        log.warning(f"Could not resolve {'cropout' if cropout else 'cutout'} image from mask: {cutout_mask_path}")
        log.warning(f"Expected {'cropout' if cropout else 'cutout'} image path: {cand}")
        return None
    

    # ---------- Tag normalization ----------

    def normalize_final_tag(self, user_tags: List[str]) -> Optional[str]:
        """
        Map arbitrary user tags to one *final* canonical tag.
        Priority: approved > needsfix > defer > reject.
        """
        tags = [t.strip().lower() for t in (user_tags or []) if t]
        matches = set()
        for canonical, keyword in self.canonical_map.items():
            kw = str(keyword).lower().strip()
            if not kw:
                continue
            if any(kw in t for t in tags):
                matches.add(canonical)

        if not matches:
            return None

        for p in self.canonical_map.keys():  # final-approved, final-needsfix, final-defer, final-reject --- IGNORE ---
            if p in matches:
                return p
        # deterministic fallback
        return sorted(matches)[0]

    # ---------- Row selection ----------

    def _iter_rows_for_review(self):
        df = self.df
        # Eligible = never reviewed OR explicitly needsfix/defer
        # Eligible = never reviewed OR explicitly needsfix
        mask = (df["mask_status"].isna()) | (df["mask_status"].isin(MASK_STATUS_FOR_ELIGIBLE_REVIEW))
        if self.only_tags:
            contains = df.apply(
                lambda r: any(
                    str(r.get(col, "")).lower().find(tok.lower()) >= 0
                    for col in ["final_mask_issue_tag", "tags", "mask_status"]
                    for tok in self.only_tags
                ),
                axis=1,
            )
            mask = mask & contains

        log.info(f"Rows eligible for final review: {int(mask.sum())}")
        for _, row in df[mask].iterrows():
            yield row

    # ---------- Sample building ----------

    def _populate_metadata_fields(self, sample: fo.Sample, row: pd.Series) -> fo.Sample:
        sample["row_index"] = int(row.name)
        for k in ["cutout_id", "stem", "cutout_name"]:
            if k in row.index:
                sample[k] = str(row[k]) if pd.notna(row[k]) else ""
        return sample

    def _load_samples(self) -> List[fo.Sample]:
        samples: List[fo.Sample] = []
        missing = 0

        for row in self._iter_rows_for_review():
            mask_p = self._resolve_cutout_mask_path(row)
            img_p = self._resolve_cutout_path(row, cropout=True)

            if not mask_p or not mask_p.exists() or not img_p or not img_p.exists():
                missing += 1
                log.warning(f"[row {row.name}] Missing image/mask. img={img_p}, mask={mask_p}")
                continue

            try:
                mask = np.array(Image.open(mask_p).convert("L"), dtype=np.uint8)
                sample = fo.Sample(filepath=str(img_p))
                sample = self._populate_metadata_fields(sample, row)
                sample["latest_mask"] = fo.Segmentation(mask=mask)

                # Show prior context via tags
                carry = []
                for col in ("final_mask_issue_tag", "tags"):
                    val = row.get(col)
                    if pd.notna(val) and val:
                        carry.extend([t.strip() for t in str(val).split(",") if t.strip()])
                if carry:
                    sample.tags = sorted(set(c.lower() for c in carry))

                samples.append(sample)
            except Exception as e:
                log.warning(f"Error building sample for row {row.name}: {e}")

        if missing:
            log.info(f"Skipped {missing} rows with missing files")
        log.info(f"Prepared {len(samples)} samples for FINAL review")
        return samples

    # ---------- FiftyOne ----------

    def _create_dataset(self, samples: List[fo.Sample]) -> fo.Dataset:
        if self.dataset_name in fo.list_datasets():
            fo.delete_dataset(self.dataset_name)
        ds = fo.Dataset(self.dataset_name)
        ds.add_samples(samples)
        return ds

    def run(self) -> None:
        samples = self._load_samples()
        if not samples:
            log.info("No eligible samples to review. Exiting.")
            return

        self.dataset = self._create_dataset(samples)
        self.session = fo.launch_app(self.dataset, port=self.port)

        log.info("FiftyOne session running — close the app when done tagging.")
        try:
            self.session.wait()
        except KeyboardInterrupt:
            log.info("Session interrupted by user")
        except Exception as e:
            log.error(f"FiftyOne session error: {e}")
        finally:
            try:
                self.session.refresh()
                self.session.close()
            except Exception:
                pass

        # Post-session side effects
        self._apply_decisions_and_handoff()
        self._save_csv()

    # ---------- LTS + DB ----------

    @staticmethod
    def _lts_target_for(src: Path, base: Path) -> Path:
        """
        Decide a target path inside LTS. Currently flattens to filename.
        Customize if you want date/species/org hierarchy.
        """
        return base / src.name

    def _copy_to_lts(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    def _move_to_lts(self, row_idx: int, image_path: Path, full_mask_path: Path) -> None:
        tgt_img = self._lts_target_for(image_path, self.lts_cutouts_dir)
        tgt_msk = self._lts_target_for(full_mask_path, self.lts_masks_dir)

        # copy (safer than move), optionally remove source
        for src, dst in [(image_path, tgt_img), (full_mask_path, tgt_msk)]:
            if src.resolve() != dst.resolve():
                self._copy_to_lts(src, dst)
                if self.remove_source:
                    try:
                        os.remove(src)
                    except Exception:
                        pass

        # Store LTS paths relative to the LTS roots (clear + portable)
        short_tgt_cutout_path = tgt_img.parent.name + "/" + tgt_img.name
        short_tgt_mask_path = tgt_msk.parent.name + "/" + tgt_msk.name
        log.info(f"[row {row_idx}] Moved to LTS: {short_tgt_cutout_path}, {short_tgt_mask_path}")
        self.df.at[row_idx, "final_cutout_path"] = str(short_tgt_cutout_path)
        self.df.at[row_idx, "final_mask_path"] = str(short_tgt_mask_path)

    def _update_persistent_db(self, row_idx: int) -> None:
        """
        Update only these columns in the persistent DB:
        - bbox_xywh (TEXT)
        - det_pred_conf (REAL)
        - final_mask_path (TEXT)
        - final_cutout_path (TEXT)
        - mask_status (TEXT)
        - mask_timestamp (TEXT)
        - refine_params (TEXT)
        - mask_reviewer (TEXT)
        - tags (TEXT)
        - initial_mask_issue_tag (TEXT)
        """
        if not (self.db_path and self.db_table and self.db_pk):
            return
        if not self.db_path.exists():
            log.warning(f"Persistent DB not found: {self.db_path}")
            return
        if not (0 <= row_idx < len(self.df)):
            return

        key = self.df.at[row_idx, self.db_pk] if self.db_pk in self.df.columns else None
        if key is None or (isinstance(key, float) and pd.isna(key)):
            return

        # Pull values from CSV row (gracefully handle missing columns)
        row = self.df.iloc[row_idx]
        bbox_xywh = row.get("bbox_xywh", None)
        det_pred_conf = row.get("det_pred_conf", None)
        final_mask_path = row.get("final_mask_path", None)
        final_cutout_path = row.get("final_cutout_path", None)
        mask_status = row.get("mask_status", None)
        mask_timestamp = self.timestamp
        refine_params = row.get("refine_params", None)
        mask_reviewer = self.reviewer
        tags = row.get("tags", None)
        initial_mask_issue_tag = row.get("initial_mask_issue_tag", None)

        try:
            with sqlite3.connect(str(self.db_path)) as con:
                cur = con.cursor()

                # Ensure the six columns exist (idempotent)
                for col, typ in [
                    ("bbox_xywh", "TEXT"),
                    ("det_pred_conf", "REAL"),
                    ("final_mask_path", "TEXT"),
                    ("final_cutout_path", "TEXT"),
                    ("mask_status", "TEXT"),
                    ("mask_timestamp", "TEXT"),
                    ("refine_params", "TEXT"),
                    ("mask_reviewer", "TEXT"),
                    ("tags", "TEXT"),
                    ("initial_mask_issue_tag", "TEXT"),
                ]:
                    try:
                        cur.execute(f"ALTER TABLE {self.db_table} ADD COLUMN {col} {typ}")
                    except Exception:
                        pass  # column already exists

                # Perform the update
                cur.execute(
                    f"""
                    UPDATE {self.db_table}
                    SET bbox_xywh=?,
                        det_pred_conf=?,
                        final_mask_path=?,
                        final_cutout_path=?,
                        mask_status=?,
                        mask_timestamp=?,
                        refine_params=?,
                        mask_reviewer=?,
                        tags=?,
                        initial_mask_issue_tag=?
                        
                    WHERE {self.db_pk}=?
                    """,
                    (
                        None if pd.isna(bbox_xywh) else str(bbox_xywh),
                        None if pd.isna(det_pred_conf) else float(det_pred_conf),
                        None if pd.isna(final_mask_path) else str(final_mask_path),
                        None if pd.isna(final_cutout_path) else str(final_cutout_path),
                        None if pd.isna(mask_status) else str(mask_status),
                        None if pd.isna(mask_timestamp) else str(mask_timestamp),
                        None if pd.isna(refine_params) else str(refine_params),
                        None if pd.isna(mask_reviewer) else str(mask_reviewer),
                        None if pd.isna(tags) else str(tags),
                        None if pd.isna(initial_mask_issue_tag) else str(initial_mask_issue_tag),
                        key,
                    ),
                )
                con.commit()
        except Exception as e:
            log.warning(f"DB update failed for key={key}: {e}")

    # ---------- Decisions + handoff ----------

    def _apply_decisions_and_handoff(self) -> None:
        if not self.dataset:
            return

        timestamp = datetime.datetime.now().isoformat()

        for sample in self.dataset:
            row_idx = int(sample["row_index"]) if sample.has_field("row_index") and sample["row_index"] is not None else -1
            if not (0 <= row_idx < len(self.df)):
                continue
            current_mask_status = self.df.at[row_idx, "mask_status"] if "mask_status" in self.df.columns else None
            final_status = self.normalize_final_tag(list(sample.tags or [])) or current_mask_status

            # Single source of truth for review outcome:
            self.df.at[row_idx, "mask_status"] = final_status
            self.df.at[row_idx, "mask_reviewer"] = self.reviewer if final_status != "unreviewed" else None
            self.df.at[row_idx, "mask_timestamp"] = timestamp

            if final_status == "final-approved":
                row = self.df.iloc[row_idx]
                full_mask_p = self._resolve_fullsized_mask_path(row)
                cutout_img_p = self._resolve_cutout_path(row, cropout=False)

                if full_mask_p and cutout_img_p and full_mask_p.exists() and cutout_img_p.exists():
                    self._move_to_lts(row_idx, cutout_img_p, full_mask_p)
                else:
                    log.warning(
                        f"[row {row_idx}] could not resolve files for LTS move "
                        f"(cutout: {cutout_img_p}, full_mask: {full_mask_p})"
                    )

                # Only your six columns are touched in DB. Only updates final-approved rows.
                self._update_persistent_db(row_idx)


def main(cfg: DictConfig) -> None:
    log.info("Starting FINAL mask inspection (temp_db mode)")
    try:
        FinalMaskInspector(cfg).run()
    except Exception as e:
        log.error(f"Fatal error in final inspection: {e}", exc_info=True)
    log.info("Finished FINAL mask inspection")
