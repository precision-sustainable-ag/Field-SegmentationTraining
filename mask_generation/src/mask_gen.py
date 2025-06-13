
import json
import logging
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
# Configure logging
log = logging.getLogger(__name__)

def process_missing_red(cfg: DictConfig) -> np.array:
    """Doing something here"""
    

def process_missing_white(cfg: DictConfig) -> np.array:
    """Doing something here"""
    pass

def process_mat_present(cfg: DictConfig) -> np.array:
    """Doing something here"""
    pass



def main(cfg: DictConfig) -> None:
    """
    Entry point for running the weed detection process using configuration settings.

    Args:
        cfg (DictConfig): Configuration object containing model path and image directories.
    """
    # read db

    # load images

    # for image in images:
    #     if image is missing red:
    #        process_missing_red(cfg)
    #     elif image is missing white:
    #        process_missing_white(cfg)


    # Save mask results

    log.info("Weed detection process completed.")