import os
import yaml
import torch
import logging
import numpy as np

from tqdm import tqdm
from utils.unet import UNet
from torch import optim, nn
from datetime import datetime
from omegaconf import DictConfig
from src.utils.early_stopping_train import EarlyStopping
import segmentation_models_pytorch as smp
from src.utils.custom_dataset import CustomDataset
from torch.utils.data import DataLoader, random_split
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

# Configure logging
log = logging.getLogger(__name__)

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class TrainUNetSegmentation:
    """
    U-Net training pipeline class.

    This class encapsulates the workflow for training a U-Net model,
    including dataset preparation, model initialization, and evaluation.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the U-Net training pipeline.

        Args:
            cfg (DictConfig): Configuration object containing parameters 
                            such as learning rate, batch size, and paths.
        """
        log.info("Initializing U-Net training...")
        # Load configuration parameters
        self.learning_rate = cfg.unet_conf.learning_rate
        self.batch_size = cfg.unet_conf.batch_size
        self.epochs = cfg.unet_conf.epochs
        self.model_save_dir = cfg.paths.model_save_dir
        self.data_dir = cfg.paths.data_dir

        # Create date-based directories for saving outputs
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.current_date_dir = os.path.join(self.model_save_dir, f"runs_{current_date}")
        self.project_dir = os.path.join(self.current_date_dir, "project")
        self.weights_save_dir = os.path.join(self.project_dir, "weights")
        self.model_save_path = os.path.join(self.weights_save_dir, "unet_segmentation.pth")

        os.makedirs(self.current_date_dir, exist_ok=True)
        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.weights_save_dir, exist_ok=True)

        self._load_data() # Load dataset and create train/val splits
        self._build_model() # Initialize model, optimizer, loss function, and metrics
        self._setup_metric_logging() # Initialize CSV file for logging metrics

        log.info("Initialization complete.")

    def _load_data(self):
        """Loads the dataset and creates training/validation splits."""
        log.info("Loading dataset...")
        dataset = CustomDataset(self.data_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        log.info(f"Dataset loaded: {train_size} train samples, {val_size} validation samples.")

    def _build_model(self):
        """Initializes the U-Net model, optimizer, loss function, and metrics."""
        self.model = UNet(in_channels=3, num_classes=1).to(device)

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.mean_iou = MeanIoU(num_classes=2).to(device)
        self.generalized_dice = GeneralizedDiceScore(num_classes=2).to(device)

    def _setup_metric_logging(self):
        """Initializes the CSV file for logging training metrics."""
        self.metrics_log_path = os.path.join(self.project_dir, "results.csv")
        with open(self.metrics_log_path, "w") as log_file:
            log_file.write("Epoch\tTrain Loss\tVal Loss\tIoU\tGeneralized Dice Score\n")
        log.info(f"Metrics will be logged to {self.metrics_log_path}")


    def _train_one_epoch(self, epoch: int):
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch}", leave=False): # batch is a tuple of (images, masks) 
            inputs, targets = batch[0].to(device), batch[1].to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def _validate(self, epoch: int):
        """
        Validates the model on the validation dataset and computes metrics.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple: Validation loss, IoU, Dice Score.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds, all_targets = [], []

        # Reset metrics
        self.mean_iou.reset()
        self.generalized_dice.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", leave=False): # batch is a tuple of (images, masks) 
                inputs, targets = batch[0].float().to(device), batch[1].float().to(device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item()

                # Convert predictions to binary and ensure targets are binary integers
                preds = (torch.sigmoid(outputs) > 0.5).long()
                targets = targets.long()

                # Update torchmetrics
                self.mean_iou.update(preds, targets)
                self.generalized_dice.update(preds, targets)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        # Compute metrics
        log.info("Computing mean_iou_score...")
        mean_iou_score = self.mean_iou.compute().item()
        log.info("Computing dice_score...")        
        dice_score = self.generalized_dice.compute().item()

        log.info(f"IoU: {mean_iou_score}, Dice: {dice_score}")
        return running_loss / len(self.val_loader), mean_iou_score, dice_score

    def _log_epoch_metrics(self, epoch, train_loss, val_loss, iou, dice):
        """
        Logs training and validation metrics for the current epoch.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss.
            val_loss (float): Validation loss.
            iou (float): Mean IoU.
            dice (float): Dice Score.
        """
        log.info("Logging training and validation metrics for the current epoch")
        log.info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"IoU={iou:.4f}, Dice={dice:.4f}"
        )

        # Append metrics to the log file
        with open(self.metrics_log_path, "a") as log_file:
            log_file.write(
                f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{iou:.4f}\t{dice:.4f}\n"
            )

    def _add_augmentation_transforms_dict_to_unet_conf(self):
        """
        Add augmentation transforms dictionary to the UNet configuration file.
        """
        # Get augmentation transforms applied during training
        dataset = CustomDataset(root_path=str(self.data_dir))
        augmentation_dict = dataset.get_transforms_dict()

        # Add augmentation transforms to unet_conf.yaml
        unet_conf_path = self.results_dir_with_timestamp.parent / "unet_conf.yaml"
        with open(unet_conf_path, 'r') as file:
            unet_conf = yaml.load(file, Loader=yaml.FullLoader)
            unet_conf.update({"augmentations":augmentation_dict})

        with open(unet_conf_path, 'w') as file:
            yaml.dump(unet_conf, file, sort_keys=False)

    def _dataset_metrics(self, cfg: DictConfig):
        """
        Logs dataset information to a JSON file.
        """
        log.info("Logging the dataset information to a JSON file.")

        # Get augmentation transforms applied during training
        augmentation_dict = CustomDataset(root_path=str(self.data_dir)).get_transforms_dict()

        dataset_info = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "augmentations": augmentation_dict
        }

        dataset_save_path = os.path.join(self.project_dir, "unet_conf.yaml")
        with open(dataset_save_path, "w") as dataset_file:
            yaml.dump(dataset_info, dataset_file, sort_keys=False)

    def train_and_validate_multiple_epochs(self):
        """
        Trains the U-Net model over multiple epochs and evaluates after each epoch.
        """
        log.info("Starting training...")
        early_stopper = EarlyStopping(patience=5, min_delta=0.001)
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss, iou, dice = self._validate(epoch)

            # Save model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"✅ Saved best model at epoch {epoch}")

            # Early stopping check
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

            self._log_epoch_metrics(epoch, train_loss, val_loss, iou, dice)

        self._dataset_metrics(DictConfig)

        log.info("Training completed. Saving model...")

        # Save the trained model
        torch.save(self.model.state_dict(), self.model_save_path)
        log.info(f"Model saved in {self.model_save_dir}.")

def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and start training.

    Args:
        cfg (DictConfig): Configuration object.
    """
    trainer = TrainUNetSegmentation(cfg)
    trainer.train_and_validate_multiple_epochs()
