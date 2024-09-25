import os
import cv2
import uuid
import torch
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import date
from statistics import mean
from ultralytics import YOLO
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from utils.utils import mask_to_yolo_segmentation
from torch.nn.functional import threshold, normalize
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


# Configure logging
log = logging.getLogger(__name__)

# Set device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

class TrainYOLOSegmentation:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the YOLO segmentation training process.

        Args:
            cfg: Configuration object containing all necessary parameters for training.
        """
        log.info("Initializing the YOLO segmentation training process.")

        # Set the YOLO model version and data YAML file
        self.YOLO_model_version = cfg.yolo_conf.model_version
        self.data_yaml = cfg.yolo_conf.data_yaml
        self.epochs = cfg.yolo_conf.epochs
        self.batch_size = cfg.yolo_conf.batch_size
        self.img_size = cfg.yolo_conf.img_size
        self.model_export_format = cfg.yolo_conf.model_export_format
        self.model = YOLO(self.YOLO_model_version)

        # Set the paths for the images, masks and yolo format labels directories
        self.image_dir = cfg.paths.image_dir
        self.masks_dir = cfg.paths.mask_dir #make sure to convert binary masks to yolo format first

    # def convert_mask_to_yolo_format(self):
    #     """
    #     Converts mask files in the specified directory to YOLO format.
    #     This method iterates over all mask files in the directory specified by `self.masks_dir`,
    #     converts each mask to YOLO segmentation format, and saves the converted files in the
    #     directory specified by `self.yolo_format_labels`.
    #     Args:
    #         None    
    #     Returns:                
    #         None
    #     """
    #     for mask_path in os.listdir(self.masks_dir):
    #         mask_path_stem = Path(mask_path).stem
    #         yolo_format_path = os.path.join(self.yolo_format_labels, f"{mask_path_stem}.txt")
    #         mask_to_yolo_segmentation(mask_path, yolo_format_path)

    def train(self):
        """
        Trains the YOLO segmentation model using the specified parameters.
        This method initializes the training process for the YOLO segmentation model.
        It uses the configuration provided in the instance variables to set up the training.
        Args:
            None
        Returns:
            None
        """
        log.info("Starting the YOLO segmentation training process.")

        self.model.train(
            data=self.data_yaml,  
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,  
            task="segment",       
        )

    def validate(self):
        """
        Validate the YOLO segmentation model.
        This method runs the validation process on the YOLO segmentation model
        to evaluate its performance on the validation dataset.
        Args:
            None
        Returns:
            None
        """
        log.info("Starting the YOLO segmentation model validation process.")

        self.model.val()

    def save_model(self):
        """
        Saves the current model to the specified export format.
        This method exports the model using the format defined in `self.model_export_format`.
        Args:
            None
        Returns:
            None
        """
        log.info("Exporting the YOLO segmentation model.")

        self.model.export(self.model_export_format)

def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and run the YOLO segmentation training, validation, and model export.
    Args:
        cfg (DictConfig): Configuration object containing settings for the YOLO segmentation process.
    """
    # Initialize the trainer with custom settings
    trainer = TrainYOLOSegmentation(cfg)
    
    trainer.train() # Start training the model
    trainer.validate() # Validate the model
    trainer.export_model()  # Export the model

