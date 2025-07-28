import datetime
import logging
from pathlib import Path

import cv2
import json
import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from typing import Dict, Tuple, Any

from src.mask_gen_utils.mask_inspect import InspectionDB
from src.mask_gen_utils.missing_red  import MissingRed
from src.mask_gen_utils.missing_white import MissingWhite
from src.mask_gen_utils.present_mat import PresentMat

log = logging.getLogger(__name__)

class RefineMask:
    def __init__(self, cfg: DictConfig) -> None:
        self.mask_generation_dir = Path(cfg.paths.project_maskgen_dir)
        self.cutout_dir = self.mask_generation_dir / "cutouts"
        self.refined_mask_dir = self.mask_generation_dir / "refined_masks"
        self.refined_mask_dir.mkdir(parents=True, exist_ok=True)

        refine_cfg = cfg.mask_gen.refine
        self.processors = {
            "missing_red": (MissingRed(refine_cfg.missing_red), refine_cfg.missing_red),
            "missing_white": (MissingWhite(refine_cfg.missing_white), refine_cfg.missing_white),
            "present_mat": (PresentMat(refine_cfg.present_mat), refine_cfg.present_mat),
        }

        self.db_path = cfg.paths.agir_field_db
        self.db = InspectionDB(self.db_path)

        self.timestamp = datetime.datetime.now().isoformat()

        self.only_tags = cfg.mask_gen.refine.only_tags

    def _process_single_image(self, image_path: str, mask_path: str, tag: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        log.info(f"Refining mask for: {Path(image_path).name}")

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        processor_info = self.processors.get(tag)
        if processor_info is None:
            log.error(f"Unknown tag '{tag}' for image {Path(image_path).name}. Skipping.")
            return None, None

        processor, cfg = processor_info
        log.info(f"Processing tag: {tag}")
        refined_mask = processor.process(image, mask)
        return refined_mask, cfg


    def _save_refined_mask(self, mask: np.ndarray, image_name: str) -> None:
        output_path = self.refined_mask_dir / image_name.replace(".jpg", "_mask.png")
        cv2.imwrite(str(output_path), mask)
        log.info(f"Saved refined mask: {output_path}")

    def process_cutout_dir(self) -> None:
        log.info("Loading images needing refinement from database...")
        images = self.db.get_images_for_refinement(only_tags=self.only_tags)
        updates = []
        for _, image_path, mask_path, _, initial_tag, final_tag, tags, _, _, _, _ in images:
            tag = initial_tag.split(",")[0].strip().lower()
            image_name = Path(image_path).name
            refined_mask, refine_cfg = self._process_single_image(image_path, mask_path, tag)
            self._save_refined_mask(refined_mask, image_name)
            status = "refined"
            reviewer = "matt"
            timestamp = self.timestamp
            refine_params_str = json.dumps(OmegaConf.to_container(refine_cfg, resolve=True))
            updates.append((initial_tag, final_tag, tags, status, reviewer, timestamp, refine_params_str, image_name))

        self.db.bulk_update_tags(updates)
        self.db.commit()
        log.info("All images processed and updated in the database.")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    refine_mask = RefineMask(cfg)
    refine_mask.process_cutout_dir()
    log.info("Refining mask process completed successfully.")

if __name__ == "__main__":
    main()