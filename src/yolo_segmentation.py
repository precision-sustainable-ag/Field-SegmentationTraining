import torch
import logging
from ultralytics import YOLO
from omegaconf import DictConfig


# Configure logging
log = logging.getLogger(__name__)

# Set device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

class TrainYOLOSegmentation:
    """
    Class to handle the training, validation, and exporting of a YOLO segmentation model for AgIR.

    This class abstracts the process of training a YOLO model for segmentation tasks. 
    It provides methods for initializing the model, training, validating, and exporting 
    the trained model in a specified format. The configuration required for these operations 
    is passed during the initialization of the class.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize the YOLO segmentation training process.

        Args:
            cfg: Configuration object containing all necessary parameters for training.
        """
        log.info("Initializing the YOLO segmentation training process.")

        # Set the YOLO model version and data YAML file
        self.YOLO_model_version = cfg.yolo_conf.model_version
        self.data_yaml = cfg.yolo_conf.train_config
        self.epochs = cfg.yolo_conf.epochs
        self.batch_size = cfg.yolo_conf.batch_size
        self.img_size = cfg.yolo_conf.image_size
        self.model_export_format = cfg.yolo_conf.model_format
        self.model = YOLO(self.YOLO_model_version)
        self.single_cls = cfg.yolo_conf.single_cls

        # Print the configuration or attributes
        log.info(f"YOLO Model Version: {self.YOLO_model_version}")
        log.info(f"Data YAML: {self.data_yaml}")
        log.info(f"Epochs: {self.epochs}")
        log.info(f"Batch Size: {self.batch_size}")
        log.info(f"Image Size: {self.img_size}")
        log.info(f"Model Export Format: {self.model_export_format}")


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
            single_cls=True,       
        )

        log.info("Finished training the YOLO segmentation model.")

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

        log.info("Finished validating the YOLO segmentation model.")

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
    log.info("Starting the YOLO segmentation training process.")
    trainer.train() # Start training the model
    log.info("Starting the YOLO segmentation model validation process.")
    trainer.validate() # Validate the model
    log.info("Exporting the YOLO segmentation model.")
    trainer.export_model()  # Export the model

