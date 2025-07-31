# src/preprocess.py

import logging
import warnings
import traceback
from pathlib import Path

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.pipeline_log import PipelineLogger
from src.preprocess_utils.pad_gridcrop_resize import pad_gridcrop_resize
from src.preprocess_utils.train_val_test_split import train_val_test_split
from src.preprocess_utils.data_stats import compute_rgb_mean_std

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
        cutouts_dir = Path(cfg.paths.mask_gen_cutout_dir)
        masks_dir   = Path(cfg.paths.refined_masks_dir)
        # Output into preprocess directory
        out_base    = Path(cfg.paths.project_preprocess_dir)

        if cfg.tasks.preprocess.pad_gridcrop_resize:
            pad_gridcrop_resize(
                cutouts_dir=cutouts_dir,
                masks_dir=masks_dir,
                out_images=out_base/"images",
                out_masks=out_base/"masks",
                cfg=cfg.preprocess.pad_gridcrop_resize,
            )

        if cfg.tasks.preprocess.train_val_test_split:
            train_val_test_split(
                images_dir=out_base/"images",
                masks_dir=out_base/"masks",
                cfg=cfg,
            )
        
        if cfg.tasks.preprocess.compute_data_stats:
            # Compute & save RGB mean/std into ${paths.project_datastats_dir}/rgb_mean_std.json
            compute_rgb_mean_std(cfg)

        success = True

    except Exception as e:
        log.error(f"Error in preprocess: {e}")
        log.debug(traceback.format_exc())
        pipe_logger.add_error(e)
    finally:
        pipe_logger.finalize(success=success)

if __name__ == "__main__":
    preprocess()