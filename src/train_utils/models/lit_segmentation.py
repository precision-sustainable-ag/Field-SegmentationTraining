# src/train_utils/models/lit_segmentation.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Tuple, List, Dict

import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex as IoU, BinaryF1Score as Dice

class LitSegmentation(pl.LightningModule):
    """
    PyTorch Lightning module for semantic segmentation using segmentation_models.pytorch (SMP).
    
    Supports:
    - Binary segmentation (BCEWithLogitsLoss)
    - IoU and Dice metrics (batch-aggregated)
    - Dynamic model selection via Hydra config
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the Lightning module with model, loss, metrics, and hyperparameters.

        Args:
            cfg (DictConfig): Hydra configuration dictionary.
        """
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # === Prepare kwargs for SMP model ===
        model_kwargs: dict[str, Any] = {
            "arch": cfg.model.arch_name,
            "encoder_name": cfg.model.encoder_name,
            "encoder_weights": cfg.model.encoder_weights,
            "in_channels": cfg.model.in_channels,
            "classes": cfg.model.classes,
        }

        # Add optional config values if they exist
        if getattr(cfg.model, "decoder_attention_type", None):
            model_kwargs["decoder_attention_type"] = cfg.model.decoder_attention_type

        if getattr(cfg.model, "encoder_freeze", False):
            model_kwargs["encoder_freeze"] = True

        # === Instantiate the model ===
        self.model = smp.create_model(**model_kwargs)

        # === Loss ===
        self.loss_fn = nn.BCEWithLogitsLoss()

        # === Metrics ===
        self.train_iou = IoU()
        self.val_iou   = IoU()
        self.train_dice = Dice()
        self.val_dice   = Dice()

        # === Learning rate ===
        self.lr = cfg.train.optimizer.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Logits of shape (B, 1, H, W)
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for one batch.

        Args:
            batch (Tuple): (images, masks)
            batch_idx (int): Batch index

        Returns:
            Tensor: Loss value
        """
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.train_iou.update(preds, masks.long())
        self.train_dice.update(preds, masks.long())

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/iou",   self.train_iou,   on_step=False, on_epoch=True)
        self.log("train/dice",  self.train_dice,  on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for one batch.

        Args:
            batch (Tuple): (images, masks)
            batch_idx (int): Batch index

        Returns:
            Tensor: Loss value
        """
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.val_iou.update(preds, masks.long())
        self.val_dice.update(preds, masks.long())

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/iou",  self.val_iou,  on_step=False, on_epoch=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and learning rate scheduler using Hydra-instantiated config.

        Returns:
            A dictionary with keys:
            - "optimizer": instantiated optimizer
            - "lr_scheduler": dictionary with scheduler, interval, and frequency settings
        """
        optimizer = instantiate(self.cfg.train.optimizer, params=self.parameters())
        scheduler = instantiate(self.cfg.train.scheduler, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",     # or "step" if using OneCycleLR, etc.
                "frequency": 1
            }
        }
