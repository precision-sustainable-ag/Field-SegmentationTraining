import csv
import logging
import sqlite3
from typing import List, Set, Tuple

from omegaconf import DictConfig

# Logging configuration
log = logging.getLogger(__name__)

# Define your schema and column order explicitly
FIELD_DATA_COLUMNS = [
    'index',
    'Name',
    'UploadDateTimeUTC',
    'MasterRefID',
    'ImageURL',
    'CameraInfo_DateTime',
    'SizeMiB',
    'ImageIndex',
    'UsState',
    'PlantType',
    'CloudCover',
    'GroundResidue',
    'GroundCover',
    'Username',
    'CoverCropFamily',
    'GrowthStage',
    'CottonVariety',
    'CropOrFallow',
    'CropTypeSecondary',
    'Species',
    'Height',
    'SizeClass',
    'FlowerFruitOrSeeds',
    'BaseName',
    'Extension',
    'HasMatchingJpgAndRaw',
    'BatchID',
    'SubBatchIndex',
    'Stem'
]

FIELD_DATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS field_data (
    index TEXT,
    Name TEXT PRIMARY KEY,
    UploadDateTimeUTC TEXT,
    MasterRefID TEXT,
    ImageURL TEXT,
    CameraInfo_DateTime TEXT,
    SizeMiB REAL,
    ImageIndex REAL,
    UsState TEXT,
    PlantType TEXT,
    CloudCover TEXT,
    GroundResidue TEXT,
    GroundCover TEXT,
    Username TEXT,
    CoverCropFamily TEXT,
    GrowthStage TEXT,
    CottonVariety TEXT,
    CropOrFallow TEXT,
    CropTypeSecondary TEXT,
    Species TEXT,
    Height TEXT,
    SizeClass TEXT,
    FlowerFruitOrSeeds TEXT,
    BaseName TEXT,
    Extension TEXT,
    HasMatchingJpgAndRaw TEXT,
    BatchID TEXT,
    SubBatchIndex TEXT,
    Stem TEXT
)
"""

class InspectionDB:
    def __init__(self, cfg: DictConfig) -> None:
        db_path = cfg.paths.agir_field_db
        self.table_name = cfg.db.table_name
        log.info(f"Connecting to SQLite DB at {db_path}")
        try:
            self.conn = sqlite3.connect(db_path)
            self._init_db()
            self.create_and_load_metadata_table(
                metadata_table="field_data",
                schema=FIELD_DATA_SCHEMA,
                csv_file=cfg.paths.pesistent_table_data_path,
                columns=FIELD_DATA_COLUMNS
            )
            self.ensure_species_column()
            self.populate_species_from_metadata(metadata_table="field_data")
        except Exception as e:
            log.exception(f"Failed to connect or initialize DB at {db_path}: {e}")
            raise

    def _init_db(self) -> None:
        c = self.conn.cursor()
        try:
            c.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                image_id TEXT PRIMARY KEY,
                image_path TEXT,
                mask_path TEXT,
                refined_mask_path TEXT,
                relabeled_mask_path TEXT,
                initial_tag TEXT,
                final_tag TEXT,
                tags TEXT,
                status TEXT,
                reviewer TEXT,
                timestamp TEXT,
                refine_params TEXT,
                species TEXT
            )
            ''')
            
            # Add refine_params if missing
            c.execute(f"PRAGMA table_info({self.table_name})")
            existing_cols = [row[1] for row in c.fetchall()]
            if "refine_params" not in existing_cols:
                c.execute(f"ALTER TABLE {self.table_name} ADD COLUMN refine_params TEXT")

            self.conn.commit()
        except Exception as e:
            log.error(f"Error creating/updating DB table: {e}")
            raise
    
    def add_images_bulk(self, image_info_list: List[Tuple]) -> None:
        log.info(f"Bulk-inserting {len(image_info_list)} new images into DB")
        c = self.conn.cursor()
        try:
            c.executemany(f'''
                INSERT OR IGNORE INTO {self.table_name} (
                    image_id, image_path, mask_path, refined_mask_path, relabeled_mask_path,
                    initial_tag, final_tag, tags, status, reviewer, timestamp, refine_params, species
                ) VALUES (?, ?, ?, ?, ?, '', '', '', 'pending', '', '', '', '')
            ''', image_info_list)
        except Exception as e:
            log.error(f"Bulk insert failed: {e}")
            raise

    def add_or_update_image(self, image_id: str, image_path: str, mask_path: str, refined_mask_path: str, relabeled_mask_path: str, species: str) -> None:
        c = self.conn.cursor()
        try:
            c.execute(f'''
                INSERT OR IGNORE INTO {self.table_name} (
                    image_id, image_path, mask_path, refined_mask_path, relabeled_mask_path,
                    initial_tag, final_tag, tags, status, reviewer, timestamp, refine_params, species
                ) VALUES (?, ?, ?, ?, ?, '', '', '', 'pending', '', '', '', '')
            ''', (image_id, image_path, mask_path, refined_mask_path, relabeled_mask_path, species))
        except Exception as e:
            log.error(f"Insert failed for {image_id}: {e}")

    def get_images_for_review(self, only_tags: List[str] = None) -> List[Tuple]:
        c = self.conn.cursor()
        try:
            where_clauses = []
            params = []

            if only_tags:
                tag_clauses = []
                for tag in only_tags:
                    tag_clauses.append("initial_tag LIKE ?")
                    params.append(f"%{tag}%")
                where_clauses.append("(" + " OR ".join(tag_clauses) + ")")

            base_query = f"SELECT * FROM {self.table_name}"
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)

            c.execute(base_query, params)
            rows = c.fetchall()
            return rows
        except Exception as e:
            log.error(f"Error fetching images from DB: {e}")
            return []

    def bulk_update_tags(self, updates: list) -> None:
        """
        updates: list of tuples:
        (initial_tag, final_tag, tags, status, reviewer, timestamp, image_id)
        """
        log.info(f"Bulk updating tags for {len(updates)} images")
        c = self.conn.cursor()
        try:
            c.executemany(
                f"UPDATE {self.table_name} SET initial_tag=?, final_tag=?, tags=?, status=?, reviewer=?, timestamp=?, refine_params=? WHERE image_id=?",
                updates
            )
        except Exception as e:
            log.error(f"Bulk tag update failed: {e}")
            raise

    def get_all_image_ids(self) -> Set[str]:
        c = self.conn.cursor()
        try:
            c.execute(f"SELECT image_id FROM {self.table_name}")
            return set(row[0] for row in c.fetchall())
        except Exception as e:
            log.error(f"Error fetching image IDs from DB: {e}")
            return set()

    def get_images_for_refinement(self, only_tags: List[str] = None) -> List[Tuple]:
        # Only return those with a non-empty initial_tag and not already final_tag == "good"
        c = self.conn.cursor()
        try:
            base_query = f'''
                SELECT *
                FROM {self.table_name}
                WHERE 
                    (final_tag IS NULL OR final_tag != "good")
                    AND (initial_tag IS NOT NULL AND TRIM(initial_tag) != "")
            '''
            params = []
            if only_tags:
                # Compose a WHERE clause for tags using LIKE, for safety and flexibility
                tag_clauses = []
                for tag in only_tags:
                    tag_clauses.append("initial_tag LIKE ?")
                    params.append(f"%{tag}%")
                tag_filter = " AND (" + " OR ".join(tag_clauses) + ")"
                base_query += tag_filter
            c.execute(base_query, params)
            rows = c.fetchall()
            return rows
        except Exception as e:
            log.error(f"Error fetching images for refinement: {e}")
            return []
    
    def close(self):
        try:
            self.conn.close()
        except Exception as e:
            log.error(f"Error closing DB: {e}")

    def __enter__(self):
        return self
    
    def commit(self):
        log.info("Committing DB transaction.")
        try:
            self.conn.commit()
        except Exception as e:
            log.error(f"DB commit failed: {e}")
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_images_for_relabel(self, only_tags: List =None) -> List[Tuple]:
        c = self.conn.cursor()
        try:
            base_query = f"""
                SELECT * 
                FROM {self.table_name}
                WHERE (
                    final_tag IS NULL
                    OR TRIM(final_tag) = ''
                    OR LOWER(final_tag) = 'none'
                    OR LOWER(final_tag) NOT LIKE '%good%'
                )
            """
            params = []
            if only_tags:
                print(f"Filtering for tags: {only_tags}")
                tag_clauses = []
                for tag in only_tags:
                    tag_clauses.append("initial_tag LIKE ?")
                    params.append(f"%{tag}%")
                tag_filter = " AND (" + " OR ".join(tag_clauses) + ")"
                base_query += tag_filter
            c.execute(base_query, params)
            rows = c.fetchall()
            return rows
        except Exception as e:
            log.error(f"Error fetching images for relabel from DB: {e}")
            return []
        
    def ensure_species_column(self) -> None:
        """
        Ensures the 'species' column exists in the table.
        """
        log.info(f"Ensuring 'species' column exists in {self.table_name}")
        c = self.conn.cursor()
        c.execute(f"PRAGMA table_info({self.table_name})")
        cols = [row[1].lower() for row in c.fetchall()]
        if "species" not in cols:
            log.info(f"Adding column 'species' to {self.table_name}")
            c.execute(f"ALTER TABLE {self.table_name} ADD COLUMN species TEXT")
            self.conn.commit()
        else:
            log.info(f"'species' column already exists in {self.table_name}")

    def populate_species_from_metadata(self, metadata_table: str) -> None:
        """
        Populates the species column using stem-based, case-insensitive matching with another table (usually field_data or image_metadata).
        """
        log.info(f"Populating 'species' column in {self.table_name} from {metadata_table}")
        c = self.conn.cursor()
        log.info(f"Populating 'species' in {self.table_name} from '{metadata_table}' using stem-based, case-insensitive matching.")
        update_sql = f"""
        UPDATE {self.table_name}
        SET species = (
            SELECT Species FROM {metadata_table}
            WHERE 
                LOWER(SUBSTR({metadata_table}.Name, 1, INSTR({metadata_table}.Name, '.')-1)) =
                LOWER(
                    CASE
                        WHEN INSTR({self.table_name}.image_id, '_') > 0
                            THEN SUBSTR({self.table_name}.image_id, 1, INSTR({self.table_name}.image_id, '_')-1)
                        ELSE SUBSTR({self.table_name}.image_id, 1, INSTR({self.table_name}.image_id, '.')-1)
                    END
                )
        )
        WHERE image_id IS NOT NULL
        """
        c.execute(update_sql)
        self.conn.commit()
        log.info(f"Species column populated in {self.table_name}.")

    def create_and_load_metadata_table(self, metadata_table: str, schema: str, csv_file: str, columns: List[str]) -> None:
        c = self.conn.cursor()
        log.info(f"Creating/loading {metadata_table} from {csv_file}")
        c.execute(schema)
        self.conn.commit()
        # Prepare insert
        insert_sql = f"""
        INSERT OR REPLACE INTO {metadata_table} ({','.join(columns)})
        VALUES ({','.join(['?'] * len(columns))})
        """
        with open(csv_file, newline='', encoding="utf-8") as csvfile_:
            reader = csv.DictReader(csvfile_)
            rows = [tuple(row.get(col, None) for col in columns) for row in reader]
            c.executemany(insert_sql, rows)
        self.conn.commit()
        log.info(f"Loaded {len(rows)} rows into {metadata_table}")

    @staticmethod
    def get_csv_column_names(csv_file: str) -> list:
        """
        Returns the column names (header row) from a CSV file.
        """
        with open(csv_file, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            return [col.strip() for col in header]
