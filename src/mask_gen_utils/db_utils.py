import logging
import sqlite3
from typing import List, Set, Tuple

from omegaconf import DictConfig

# Logging configuration
log = logging.getLogger(__name__)

class InspectionDB:
    def __init__(self, cfg: DictConfig) -> None:
        db_path = cfg.paths.agir_field_db
        self.table_name = cfg.db.table_name
        log.info(f"Connecting to SQLite DB at {db_path}")
        try:
            self.conn = sqlite3.connect(db_path)
            self._init_db()
        except Exception as e:
            log.error(f"Failed to connect or initialize DB at {db_path}: {e}")
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
                initial_tag TEXT,
                final_tag TEXT,
                tags TEXT,
                status TEXT,
                reviewer TEXT,
                timestamp TEXT,
                refine_params TEXT
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
                    image_id, image_path, mask_path, refined_mask_path, 
                    initial_tag, final_tag, tags, status, reviewer, timestamp
                ) VALUES (?, ?, ?, ?, '', '', '', 'pending', '', '')
            ''', image_info_list)
        except Exception as e:
            log.error(f"Bulk insert failed: {e}")
            raise

    def add_or_update_image(self, image_id: str, image_path: str, mask_path: str, refined_mask_path: str) -> None:
        c = self.conn.cursor()
        try:
            c.execute(f'''
                INSERT OR IGNORE INTO {self.table_name} (
                    image_id, image_path, mask_path, refined_mask_path, 
                    initial_tag, final_tag, tags, status, reviewer, timestamp
                )
                VALUES (?, ?, ?, ?, '', '', '', 'pending', '', '')
            ''', (image_id, image_path, mask_path, refined_mask_path))
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