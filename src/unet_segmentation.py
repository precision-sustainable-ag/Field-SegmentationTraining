import os
import yaml
import torch
import logging

from tqdm import tqdm
from utils.unet import UNet
from torch import optim, nn
from datetime import datetime
from omegaconf import DictConfig
from utils.custom_dataset import CustomDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, recall_score
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
        Initializes the training pipeline.

        Args:
            cfg (DictConfig): Configuration object containing parameters 
                              such as learning rate, batch size, and paths.
        """
        log.info("Initializing U-Net training...")
        self._init_config(cfg)
        self._init_data(cfg)
        self._init_model(cfg)

        # Initialize metrics log file
        self.metrics_log_path = os.path.join(self.project_dir, "results.csv")
        with open(self.metrics_log_path, "w") as log_file:
            log_file.write("Epoch\tTrain Loss\tVal Loss\tAccuracy\tRecall\tIoU\tGeneralized Dice Score\n")
        log.info(f"Metrics will be logged to {self.metrics_log_path}")

        log.info("Initialization complete.")

    def _init_config(self, cfg: DictConfig):
        """
        Sets configuration parameters.

        Args:
            cfg (DictConfig): Configuration object.
        """
        self.learning_rate = cfg.unet_conf.learning_rate
        self.batch_size = cfg.unet_conf.batch_size
        self.epochs = cfg.unet_conf.epochs
        self.model_save_dir = cfg.paths.model_save_dir

        # Create directory with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.current_date_dir = os.path.join(self.model_save_dir, f"runs_{current_date}")
        os.makedirs(self.current_date_dir, exist_ok=True)

        # Create directory for current project
        self.project_dir = os.path.join(self.current_date_dir, "project")
        os.makedirs(self.project_dir, exist_ok=True)

        # Create directory for trained weights
        self.weights_save_dir = os.path.join(self.project_dir, "weights")
        os.makedirs(self.weights_save_dir, exist_ok=True)

        # Save model in the created directory
        self.model_save_path = os.path.join(self.weights_save_dir, "unet_segmentation.pth")

    def _init_data(self, cfg: DictConfig):
        """
        Loads the dataset and splits it into training and validation sets.

        Args:
            cfg (DictConfig): Configuration object containing dataset path.
        """
        log.info("Loading dataset...")
        dataset = CustomDataset(cfg.paths.data_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)

        # Split dataset into training and validation subsets
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        log.info(f"Dataset loaded: {train_size} train samples, {val_size} validation samples.")

    def _init_model(self, cfg: DictConfig):
        """
        Initializes the U-Net model, optimizer, loss function, and metrics.

        Args:
            cfg (DictConfig): Configuration object with model parameters.
        """
        self.model = UNet(in_channels=3, num_classes=1).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        # Initialize segmentation metrics
        self.mean_iou = MeanIoU(num_classes=2).to(device)  # Binary segmentation: 2 classes
        self.generalized_dice = GeneralizedDiceScore(num_classes=2).to(device)

    def train_val(self):
        """
        Trains the U-Net model over multiple epochs and evaluates after each epoch.
        """
        log.info("Starting training...")
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss, accuracy, recall, iou, dice = self._validate(epoch)
            self._log_epoch_metrics(epoch, train_loss, val_loss, accuracy, recall, iou, dice)

        self._dataset_metrics(DictConfig)

        log.info("Training completed. Saving model...")

        # Save the trained model
        torch.save(self.model.state_dict(), self.model_save_path)
        log.info(f"Model saved in {self.model_save_dir}.")

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
        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch}", leave=False):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
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
            tuple: Validation loss, accuracy, recall, IoU, Dice Score.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds, all_targets = [], []

        # Reset metrics
        self.mean_iou.reset()
        self.generalized_dice.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", leave=False):
                inputs, targets = batch[0].float().to(device), batch[1].float().to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
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
        accuracy = accuracy_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds, pos_label=1)
        mean_iou_score = self.mean_iou.compute().item()
        dice_score = self.generalized_dice.compute().item()

        log.info(f"Accuracy: {accuracy}, Recall: {recall}, IoU: {mean_iou_score}, Dice: {dice_score}")
        return running_loss / len(self.val_loader), accuracy, recall, mean_iou_score, dice_score

    def _log_epoch_metrics(self, epoch, train_loss, val_loss, accuracy, recall, iou, dice):
        """
        Logs training and validation metrics for the current epoch.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss.
            val_loss (float): Validation loss.
            accuracy (float): Validation accuracy.
            recall (float): Validation recall.
            iou (float): Mean IoU.
            dice (float): Dice Score.
        """
        log.info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Accuracy={accuracy:.4f}, Recall={recall:.4f}, IoU={iou:.4f}, Dice={dice:.4f}"
        )

        # Append metrics to the log file
        with open(self.metrics_log_path, "a") as log_file:
            log_file.write(
                f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{accuracy:.4f}\t"
                f"{recall:.4f}\t{iou:.4f}\t{dice:.4f}\n"
            )

    def _dataset_metrics(self, cfg: DictConfig):
        """
        Logs dataset information to a JSON file.
        """
        dataset_info = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset)
        }

        dataset_save_path = os.path.join(self.project_dir, "unet_conf.yaml")
        with open(dataset_save_path, "w") as dataset_file:
            yaml.dump(dataset_info, dataset_file)

def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and start training.

    Args:
        cfg (DictConfig): Configuration object.
    """
    trainer = TrainUNetSegmentation(cfg)
    trainer.train_val()
