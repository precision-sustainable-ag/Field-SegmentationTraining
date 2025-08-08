"""
Voxel Mask Inspection and Tagging Pipeline (temp_db version)
------------------------------------------------------------

- Loads mask entries from the shared project temp CSV.
- Creates a FiftyOne dataset for interactive review.
- Writes normalized tags and status back to the same CSV.
"""

import os
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

INSPECT_COLS = [
    "initial_mask_issue_tag",
    "final_mask_issue_tag",
    "tags",
    "mask_status",        # unreviewed | inspected | reviewed
    "mask_reviewer",
    "mask_review_datetime",
    ]

class FiftyOneMaskInspector:
    """
    Mask inspection tied to the project temp CSV (same artifact used by create_project/detect/segment).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.base_dir = Path(cfg.paths.base_dir).resolve()
        self.project_dir = Path(cfg.paths.project_maskgen_dir)
        self.temp_csv = Path(cfg.paths.project_temp_db)

        # Where segment wrote things
        self.cutouts_dir = self.project_dir / "cutouts"
        self.initial_masks_dir = self.project_dir / "initial_masks"

        # FiftyOne settings
        self.port = int(cfg.mask_gen.inspect.port)
        self.dataset_name = cfg.mask_gen.inspect.dataset_name
        self.only_tags: Optional[List[str]] = cfg.mask_gen.inspect.only_tags

        # Canonical tags
        self.canonical_mapping: Dict[str, str] = dict(cfg.mask_gen.canonical_tag_mapping)

        # Who’s reviewing
        self.reviewer = os.getenv("USER") or "unknown"

        # state
        self.dataset: Optional[fo.Dataset] = None
        self.session: Optional[fo.Session] = None

        # Load CSV once
        self.df = self._load_csv()
        self._ensure_inspect_columns()

    # ---------------- CSV I/O ----------------

    def _load_csv(self) -> pd.DataFrame:
        if not self.temp_csv.exists():
            raise FileNotFoundError(f"project_temp_db CSV not found: {self.temp_csv}")
        df = pd.read_csv(self.temp_csv)
        log.info(f"Loaded temp CSV with {len(df)} rows: {self.temp_csv}")
        return df


    def _save_csv(self) -> None:
        self.df.to_csv(self.temp_csv, index=False)
        log.info(f"Wrote updates to temp CSV: {self.temp_csv}")

    def _ensure_inspect_columns(self) -> None:
        for col in INSPECT_COLS:
            if col not in self.df.columns:
                self.df[col] = pd.Series([None] * len(self.df), dtype="object")
            else:
                # Force dtype to object so strings are fine
                self.df[col] = self.df[col].astype("object")

    # ---------------- Tag normalization ----------------

    def normalize_tags(self, user_tags: List[str]) -> List[str]:
        """
        Map arbitrary user tags -> canonical tags based on substring keywords.

        canonical_mapping example:
          { "good": "good", "bad": "bad", "other": "other", "flower": "flow", ... }
        """
        user_tags = [str(t).strip().lower() for t in (user_tags or []) if t]
        norm = set()
        for canonical, keyword in self.canonical_mapping.items():
            kw = str(keyword).lower().strip()
            if not kw:
                continue
            if any(kw in t for t in user_tags):
                norm.add(canonical.lower())
        return sorted(norm)

    # ---------------- Sample loading ----------------

    def _row_to_paths(self, row: pd.Series) -> Optional[Dict[str, Path]]:
        """
        Resolve image (cutout) and mask paths for a row. Returns None if not usable.
        Expects columns produced by `segment`:
          - initial_cutout_mask_path (preferred)
          - cutout_name (fallback to cutouts/<cutout_name>)
        """
        mask_path = None
        if pd.notna(row.get("initial_cutout_mask_path", None)):
            mask_path = Path(row["initial_cutout_mask_path"])
            # rebase to repo if relative
            if not mask_path.is_absolute():
                mask_path = (self.base_dir / mask_path).resolve()
        else:
            # final fallback: infer from cutout_name
            cname = row.get("cutout_name", None)
            if pd.isna(cname):
                return None
            mask_path = (self.cutouts_dir / cname.replace(".jpg", "_mask.png")).resolve()

        if not mask_path.exists():
            return None

        # image (cutout) path
        if pd.notna(row.get("cutout_name", None)):
            image_path = (self.cutouts_dir / str(row["cutout_name"])).resolve()
        else:
            # infer from mask name
            image_path = Path(str(mask_path).replace("_mask.png", ".jpg"))

        if not image_path.exists():
            # sometimes crop image might be PNG
            alt = Path(str(mask_path).replace("_mask.png", ".png"))
            image_path = alt if alt.exists() else image_path

        if not image_path.exists():
            # as a last resort, try the full-frame image
            # not ideal for inspection, but keeps row from being dropped
            if pd.notna(row.get("local_developed_image_path", None)):
                image_path = (self.base_dir / str(row["local_developed_image_path"])).resolve()

        if not image_path.exists():
            return None

        return {"image": image_path, "mask": mask_path}
    
    def _populate_sample_fields(self, sample: fo.Sample, row: pd.Series) -> fo.Sample:
        def _val(v):
            return "" if (v is None or pd.isna(v)) else v

        fields = [
            "det_pred_conf","app_species","upload_datetime_utc","camera_datetime",
            "image_index","us_state","plant_type","cloud_cover","ground_residue",
            "ground_cover","cover_crop_family","growth_stage","cotton_variety",
            "crop_or_fallow","crop_type_secondary","size_class","flower_fruit_or_seeds",
            "growth_habit","duration","taxonomic_genus","taxonomic_family",
            "taxonomic_order","taxonomic_subclass","taxonomic_group",
        ]
        for f in fields:
            name = f if not f.startswith("taxonomic_") else f.split("taxonomic_")[1]
            sample[name if f.startswith("taxonomic_") else f] = _val(row.get(f))
        sample["row_index"] = int(row.name)
        return sample

    def _iter_rows_for_review(self):
        """
        Filter rows to review. If `only_tags` is provided, restrict to rows whose
        final_mask_issue_tag (or mask_status) match that condition. Otherwise,
        default to anything not yet reviewed.
        """
        df = self.df

        # By default: anything not final-reviewed
        mask = (df["mask_status"].isna()) | (df["mask_status"].isin(["unreviewed", "inspected"]))
        if self.only_tags:
            # Example semantics:
            #   only_tags: ["unreviewed"] or ["good","bad"] etc.
            mask = mask & (
                df["final_mask_issue_tag"].isin(self.only_tags) |
                df["mask_status"].isin(self.only_tags) |
                df["initial_mask_issue_tag"].isin(self.only_tags) |
                df["tags"].fillna("").str.contains("|".join(self.only_tags), case=False)
            )

        for _, row in df[mask].iterrows():
            yield row

    def _load_samples(self) -> List[fo.Sample]:
        """
        Build FiftyOne samples from the CSV rows we plan to review.
        """
        samples: List[fo.Sample] = []
        count_missing = 0

        for row in self._iter_rows_for_review():
            paths = self._row_to_paths(row)
            if not paths:
                count_missing += 1
                continue

            try:
                # Initial mask
                init_mask = np.array(Image.open(paths["mask"]).convert("L"), dtype=np.uint8)
                sample = fo.Sample(filepath=str(paths["image"]))
                sample = self._populate_sample_fields(sample, row)
                sample["initial_mask"] = fo.Segmentation(mask=init_mask)

                # Propagate currently known tags (display-only)
                display_tags = []
                if pd.notna(row.get("final_mask_issue_tag", None)):
                    display_tags = [t.strip() for t in str(row["final_mask_issue_tag"]).split(",") if t.strip()]
                elif pd.notna(row.get("initial_mask_issue_tag", None)):
                    display_tags = [t.strip() for t in str(row["initial_mask_issue_tag"]).split(",") if t.strip()]
                elif pd.notna(row.get("tags", None)):
                    display_tags = [t.strip() for t in str(row["tags"]).split(",") if t.strip()]

                if display_tags:
                    sample.tags = display_tags

                # Optionally attach refined mask when you start writing those paths to CSV
                if pd.notna(row.get("refined_cutout_mask_path", None)):
                    rpath = Path(row["refined_cutout_mask_path"])
                    if not rpath.is_absolute():
                        rpath = (self.base_dir / rpath).resolve()
                    if rpath.exists():
                        refined = np.array(Image.open(rpath).convert("L"), dtype=np.uint8)
                        sample["refined_mask"] = fo.Segmentation(mask=refined)

                # Cache key so we can write back by row index later
                sample["row_index"] = int(row.name)
                samples.append(sample)

            except Exception as e:
                log.warning(f"Error building sample for row {row.name}: {e}")

        if count_missing:
            log.info(f"Skipped {count_missing} rows with missing image/mask files")

        log.info(f"Prepared {len(samples)} samples for review")
        return samples

    # ---------------- FiftyOne dataset/session ----------------

    def _create_dataset(self, samples: List[fo.Sample]) -> fo.Dataset:
        if self.dataset_name in fo.list_datasets():
            # keep it simple and replace
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

        log.info("FiftyOne session running — close the app when you're done tagging.")
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

        self._write_back_tags()
        self._save_csv()

    # ---------------- Write-back logic ----------------

    def _write_back_tags(self) -> None:
        """
        Collect tags from the FiftyOne dataset and update rows in self.df.
        """
        if not self.dataset:
            return

        timestamp = datetime.datetime.now().isoformat()

        for sample in self.dataset:
            row_idx = int(sample["row_index"]) if "row_index" in sample else -1
            if row_idx < 0 or row_idx >= len(self.df):
                continue

            # Normalize tags
            canonical = self.normalize_tags(sample.tags or [])
            tags_set = set(t.lower() for t in canonical)

            # Determine state machine
            final_tag = None
            status = None
            initial_tag = self.df.at[row_idx, "initial_mask_issue_tag"]
            initial_tag = str(initial_tag).lower() if pd.notna(initial_tag) else None

            if "good" in tags_set:
                final_tag, status = "good", "finalized"
            elif "bad" in tags_set:
                final_tag, status = "bad", "reviewed"
            elif "other" in tags_set:
                final_tag, status = "other", "reviewed"
            elif tags_set:
                # inspected but not finalized
                status = "inspected"
            else:
                status = "unreviewed"

            # Init tag: if none or stale, set to anything except terminal states
            if not initial_tag or initial_tag not in tags_set:
                non_terminal = tags_set - {"good", "bad", "other"}
                initial_tag = ",".join(sorted(non_terminal)) if non_terminal else initial_tag

            tags_str = ",".join(sorted(tags_set)) if tags_set else None


            # Write back
            self.df.at[row_idx, "initial_mask_issue_tag"] = initial_tag
            self.df.at[row_idx, "final_mask_issue_tag"] = final_tag
            self.df.at[row_idx, "tags"] = tags_str
            self.df.at[row_idx, "mask_status"] = status
            self.df.at[row_idx, "mask_reviewer"] = self.reviewer if status in ("inspected", "reviewed") else None
            self.df.at[row_idx, "mask_review_datetime"] = timestamp


def main(cfg: DictConfig) -> None:
    log.info("Starting mask inspection (temp_db mode)")
    try:
        FiftyOneMaskInspector(cfg).run()
    except Exception as e:
        log.error(f"Fatal error in inspection: {e}")
    log.info("Finished mask inspection")
