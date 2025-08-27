import logging
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

# ---------------------------------------------------------------------
# Logging (your preferred format)
# ---------------------------------------------------------------------
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config dataclass (so this file runs standalone or with Hydra)
# Expected keys (aligns with your past scripts):
#   cfg.paths.project_temp_db           -> CSV with “temp DB”
#   cfg.paths.lts_db                    -> SQLite path to LTS DB
#   cfg.paths.lts_full_masks_dir        -> destination dir for full masks
#   cfg.paths.lts_cutouts_dir           -> destination dir for cutout color PNGs
#   cfg.lts.table_name                  -> table to update in LTS DB (e.g., "field_data" or your cutout table)
#   cfg.lts.image_id_column             -> name of image_id column (e.g., "image_id")
#   cfg.finalize.drop_legacy_columns    -> bool (default True)
#   cfg.finalize.file_checks            -> bool (default True)  ensure src files exist before moving
#   cfg.finalize.overwrite              -> bool (default False) allow overwriting existing LTS paths
#
# Your temp CSV should have columns (if present, they’ll be used):
#   - image_id
#   - temp_relabeled_full_mask_path
#   - temp_relabeled_cutout_image_path
#   - refine_params
#   - seg_note
#   - detection_note
#   - mask_reviewer
#   - final_mask_issue_tag
#   - mask_timestamp
#   - (optionally) mask_status
# ---------------------------------------------------------------------

class FinalizeMasks:
    REQUIRED_TEMP_COLS = [
        "image_id",
        "temp_relabeled_full_mask_path",
        "temp_relabeled_cutout_image_path",
    ]

    OPTIONAL_WRITE_COLS = [
        "refine_params",
        "seg_note",
        "detection_note",
        "mask_reviewer",
        "final_mask_issue_tag",
        "mask_timestamp",
        # "mask_status" is set explicitly to 'finalized'
    ]

    DROP_LEGACY_COLS = [
        "mask_review_datetime",
    ]

    def __init__(self, cfg: DictConfig):
        # Paths
        self.project_temp_db = Path(cfg.paths.project_temp_db)
        self.lts_db = Path(cfg.paths.agir_field_db)
        self.lts_full_masks_dir = Path(cfg.paths.field_masks_dir)
        self.lts_cutouts_dir = Path(cfg.paths.field_cutouts_dir)
        # Ensure destination dirs exist
        self.lts_full_masks_dir.mkdir(parents=True, exist_ok=True)
        self.lts_cutouts_dir.mkdir(parents=True, exist_ok=True)

        # LTS
        self.lts_table_name = "field_data"
        self.lts_image_id_column = "image_id"

        # Options
        self.drop_legacy_columns = cfg.finalize.drop_legacy_columns
        self.file_checks = cfg.finalize.file_checks
        self.overwrite = cfg.finalize.overwrite

    # ------------------------------ I/O ------------------------------

    def load_temp_db(self) -> pd.DataFrame:
        temp_csv = self.project_temp_db
        assert temp_csv.exists(), f"Temp DB CSV not found: {temp_csv}"
        df = pd.read_csv(temp_csv)
        missing = [c for c in self.REQUIRED_TEMP_COLS if c not in df.columns]
        assert not missing, f"Temp DB missing required columns: {missing}"
        return df

    def _mask_is_finalized(self, row: pd.Series) -> bool:
        return row.get("mask_status") == "finalized"

    def _row_has_relabeled_mask(self, row: pd.Series) -> bool:
        full_mask = row.get("temp_relabeled_full_mask_path")
        cutout_png = row.get("temp_relabeled_cutout_image_path")
        
        if pd.isna(full_mask) or pd.isna(cutout_png):
            log.warning(f"Row missing temp paths: {row.get('image_id')}")
            return False

        return Path(str(full_mask)).exists() and Path(str(cutout_png)).exists()

    def _row_is_ready(self, row: pd.Series) -> bool:
        """
        A conservative “ready to finalize” heuristic:
        - temp_relabeled_full_mask_path present (and file exists if checks on)
        - temp_relabeled_cutout_image_path present (and file exists if checks on)
        """
        if self._mask_is_finalized(row):
            return True
        if self._row_has_relabeled_mask(row):
            return True
        return False

    def filter_rows_to_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df.apply(self._row_is_ready, axis=1)
        filtered = df[mask].copy()
        log.info(f"Rows ready to finalize: {len(filtered)}/{len(df)}")
        return filtered

    # ------------------------------ LTS moves ------------------------------

    def _dest_for_full_mask(self, image_id: str, src_path: Path) -> Path:
        # Keep extension as-is (often .png for masks)
        return self.lts_full_masks_dir / f"{image_id}_mask{src_path.suffix}"

    def _dest_for_cutout_png(self, image_id: str, src_path: Path) -> Path:
        return self.lts_cutouts_dir / f"{image_id}_0{src_path.suffix}"

    def _move_file(self, src: Path, dst: Path):
        if dst.exists():
            if self.overwrite:
                if dst.is_file():
                    log.warning(f"Overwriting existing file: {dst}")
                    dst.unlink()
                else:
                    log.warning(f"Destination exists but is not a file: {dst}. Removing directory.")
                    shutil.rmtree(dst)
            else:
                # Don’t move—assume a previous run; keep path
                log.warning(f"Destination exists (skipping move): {dst}")
                return
        dst.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)  # copy to keep provenance; change to move if desired

    def move_assets_to_lts(self, df_ready: pd.DataFrame) -> pd.DataFrame:
        """
        Copies files to LTS and returns a copy of df_ready with new path columns:
          - final_mask_path
          - cutout_image_path
        """
        df = df_ready.copy()
        final_mask_paths = []
        final_cutout_paths = []

        for _, row in df.iterrows():
            image_id = str(row["image_id"])  # Ensure image_id is a string
            image_id_stem = str(Path(image_id).stem)  # Ensure image_id is a string
            src_full_mask = Path(str(row["temp_relabeled_full_mask_path"]))
            src_cutout_png = Path(str(row["temp_relabeled_cutout_image_path"]))

            dst_full_mask = self._dest_for_full_mask(image_id_stem, src_full_mask)
            dst_cutout_png = self._dest_for_cutout_png(image_id_stem, src_cutout_png)

            # Move/copy
            if src_full_mask.exists():
                self._move_file(src_full_mask, dst_full_mask)
            else:
                log.error(f"Missing full mask file for {image_id}: {src_full_mask}")

            if src_cutout_png.exists():
                self._move_file(src_cutout_png, dst_cutout_png)
            else:
                log.error(f"Missing cutout image file for {image_id}: {src_cutout_png}")

            final_mask_paths.append(str(dst_full_mask))
            final_cutout_paths.append(str(dst_cutout_png))

        df["final_mask_path"] = final_mask_paths
        df["cutout_image_path"] = final_cutout_paths
        return df

    # ------------------------------ SQLite helpers ------------------------------

    def _connect(self):
        return sqlite3.connect(self.lts_db)

    def _column_exists(self, con: sqlite3.Connection, table: str, col: str) -> bool:
        cur = con.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        return col in cols

    def _safe_drop_column(self, con: sqlite3.Connection, table: str, col: str):
        if not self._column_exists(con, table, col):
            return
        try:
            # SQLite >= 3.35 supports DROP COLUMN
            con.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
            log.info(f"Dropped legacy column: {col}")
        except sqlite3.OperationalError as e:
            # Fallback: warn (or implement rebuild if you really need it)
            log.warning(f"Could not drop column {col} (SQLite version?). Error: {e}")

    # ------------------------------ DB update ------------------------------

    def update_lts_db(self, df_final: pd.DataFrame) -> None:
        """
        For each row:
          - SET mask_status='finalized'
          - SET mask_timestamp, refine_params, seg_note, detection_note, mask_reviewer, final_mask_issue_tag (if present)
          - SET final_mask_path, cutout_image_path (from moved locations)
          - DO NOT write temp_* columns; they’re temporary
          - Optionally DROP legacy columns
        """
        table = self.lts_table_name
        id_col = self.lts_image_id_column

        # Build list of writeable columns that exist in the LTS table
        with self._connect() as con:
            con.execute("PRAGMA foreign_keys = ON;")

            # Confirm id_col exists
            assert self._column_exists(con, table, id_col), f"{table}.{id_col} not found in LTS DB"

            # Columns we intend to write if the table has them
            candidate_cols = [
                "mask_status",
                "mask_timestamp",
                "refine_params",
                "seg_note",
                "detection_note",
                "mask_reviewer",
                "final_mask_issue_tag",
                "final_mask_path",
                "cutout_image_path",
                "bbox_xywh",
                "det_pred_conf",
                "initial_mask_issue_tag",
                "tags"
            ]

            existing_cols = [c for c in candidate_cols if self._column_exists(con, table, c)]
            missing_cols = [c for c in candidate_cols if c not in existing_cols]
            if missing_cols:
                log.warning(f"The following columns do not exist in {table} and will be ignored: {missing_cols}")

            # Prepare parameterized UPDATE
            set_clauses = []
            for c in existing_cols:
                if c == "mask_status":
                    set_clauses.append(f"{c}=?")
                else:
                    set_clauses.append(f"{c}=?")

            set_sql = ", ".join(set_clauses)
            sql = f"UPDATE {table} SET {set_sql} WHERE {id_col}=?"

            # Iterate rows
            updated = 0
            for _, row in df_final.iterrows():
                # Values must align with existing_cols order
                values = []
                for c in existing_cols:
                    if c == "mask_status":
                        values.append("finalized")
                    else:
                        values.append(row.get(c, None))
                values.append(row["image_id"])  # WHERE image_id=?

                cur = con.execute(sql, values)
                if cur.rowcount == 0:
                    log.warning(f"No LTS row matched {id_col}={row['image_id']}")
                else:
                    updated += cur.rowcount

            log.info(f"Updated {updated} rows in LTS DB: {self.lts_db}")

            if self.drop_legacy_columns:
                for col in self.DROP_LEGACY_COLS:
                    self._safe_drop_column(con, table, col)

    # ------------------------------ Orchestrator ------------------------------

    def run(self) -> None:
        # 1) Read temp DB
        df = self.load_temp_db()

        # 2) Filter rows ready to finalize (both temp files present)
        df_ready = self.filter_rows_to_finalize(df)
        # print(df_ready)
        # exit()
        
        if df_ready.empty:
            log.info("No rows ready to finalize. Nothing to do.")
            return

        # 3) Move assets to LTS and create final path columns
        df_moved = self.move_assets_to_lts(df_ready)

        # 4) Copy/update all required write columns into df_moved (if present)
        #    Ensure columns exist even if missing in temp CSV (set to None)
        for col in ["mask_timestamp", "refine_params", "seg_note", "detection_note",
                    "mask_reviewer", "final_mask_issue_tag"]:
            if col not in df_moved.columns:
                df_moved[col] = None

        # 5) Update LTS DB
        self.update_lts_db(df_moved)

        log.info("Finalization complete.")


def main(cfg: DictConfig) -> None:
    FinalizeMasks(cfg).run()
