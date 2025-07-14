# preprocess.py

import logging
import warnings
import traceback
from pathlib import Path

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.pipeline_log import PipelineLogger
from src.utils.preprocess_utils import pad_gridcrop_resize

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def preprocess(cfg: DictConfig) -> None:
    """
    Entry point for preprocessing:
    - pad/grid-crop/resize cutouts & masks to cfg.preprocess.size
    """
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    pipe_logger = PipelineLogger(run_dir, "preprocess")

    # Hook warnings into pipeline logger
    orig_showwarning = warnings.showwarning
    def _capture_warning(msg, cat, fn, ln, file=None, line=None):
        txt = warnings.formatwarning(msg, cat, fn, ln, line)
        pipe_logger.add_warning(txt)
        return orig_showwarning(msg, cat, fn, ln, file, line)
    warnings.showwarning = _capture_warning

    success = False
    try:
        # Source directories from mask_gen (use paths config)
        cutouts_dir = Path(cfg.paths.initial_mask_inspection_source)
        masks_dir   = Path(cfg.paths.refined_masks_dir)

        # Output into preprocess directory
        out_base    = Path(cfg.paths.project_preprocess_dir)
        out_images  = out_base / "images"
        out_masks   = out_base / "masks"

        # Run pad/grid-crop/resize
        pad_gridcrop_resize(
            cutouts_dir=cutouts_dir,
            masks_dir=masks_dir,
            out_images=out_images,
            out_masks=out_masks,
            cfg=cfg.preprocess,
        )

        success = True
    except Exception as e:
        log.error(f"Error in preprocess: {e}")
        log.debug(traceback.format_exc())
        pipe_logger.add_error(e)
    finally:
        pipe_logger.finalize(success=success)

if __name__ == "__main__":
    preprocess()