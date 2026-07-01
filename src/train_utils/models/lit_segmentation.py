# src/train_utils/models/lit_segmentation.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Tuple, List, Dict

import segmentation_models_pytorch as smp
from torchmetrics import MetricCollection

class LitSegmentation(pl.LightningModule):
    """
    PyTorch Lightning module for semantic segmentation using segmentation_models.pytorch (SMP).
    
    Supports:
    - Dynamic losses via Hydra config
    - Dynamic metrics (IoU, Dice, Precision, Recall, AUROC, etc.) via Hydra config
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
        # === Instantiating Multiple Losses ===
        self.losses = nn.ModuleDict()
        self.loss_weights = {}
        
        for loss_name, loss_cfg in cfg.train.losses.items():
            # Check the enabled flag before instantiating
            if loss_cfg.get("enabled", False):
                self.losses[loss_name] = instantiate(loss_cfg.loss)
                self.loss_weights[loss_name] = loss_cfg.weight
                
        # Optional: Add a safety check to ensure at least one loss is enabled
        if len(self.losses) == 0:
            raise ValueError("No loss functions are enabled in the configuration!")

        # === Dynamic Metrics Initialization ===
        metrics_dict = {}
        if hasattr(cfg, "evaluation") and hasattr(cfg.evaluation, "metrics"):
            for metric_name, metric_cfg in cfg.evaluation.metrics.items():
                if getattr(metric_cfg, "enabled", False):
                    metrics_dict[metric_name] = instantiate(metric_cfg.metric)
        else:
            raise ValueError("No evaluation metrics found in configuration.")

        # MetricCollection handles cross-GPU syncing automatically!
        # We clone the collection for each phase to isolate their internal states.
        base_metrics = MetricCollection(metrics_dict)
        
        # We assign these directly to the module so Lightning registers them
        self.train_metrics = base_metrics.clone(prefix="train/")
        self.val_metrics = base_metrics.clone(prefix="val/")
        self.test_metrics = base_metrics.clone(prefix="test/")
        
        # Separate collections for threshold-independent metrics (AUROC, PR-AUC)
        # because they require raw probabilities instead of binary predictions.
        prob_metrics_dict = {}
        bin_metrics_dict = {}
        
        for name, metric in metrics_dict.items():
             if "AUROC" in str(type(metric)) or "AveragePrecision" in str(type(metric)):
                 prob_metrics_dict[name] = metric
             else:
                 bin_metrics_dict[name] = metric
                 
        # Create discrete collections for Train, Val, and Test to avoid state collisions
        self.train_prob_metrics = MetricCollection(prob_metrics_dict).clone(prefix="train/")
        self.val_prob_metrics = MetricCollection(prob_metrics_dict).clone(prefix="val/")
        self.test_prob_metrics = MetricCollection(prob_metrics_dict).clone(prefix="test/")
        
        self.train_bin_metrics = MetricCollection(bin_metrics_dict).clone(prefix="train/")
        self.val_bin_metrics = MetricCollection(bin_metrics_dict).clone(prefix="val/")
        self.test_bin_metrics = MetricCollection(bin_metrics_dict).clone(prefix="test/")

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

        # Calculate weighted combined loss
        total_loss = 0.0
        # Ensure masks are floats for BCE and SMP losses
        masks_float = masks.float() 
        
        for name, criterion in self.losses.items():
            loss_val = criterion(logits, masks_float)
            weight = self.loss_weights[name]
            total_loss += weight * loss_val
            
            # Log individual loss components for WandB graphs
            self.log(f"train/loss_{name}", loss_val, on_step=False, on_epoch=True, sync_dist=True)
        
        self.log("train/loss_total", total_loss, on_step=False, on_epoch=True, sync_dist=True)

        # Calculate and log dynamic metrics
        # 1. Get threshold from config (default to 0.5 if not found)
        thr = getattr(self.cfg.evaluation.settings, "threshold", 0.5)
        
        # 2. Get probabilities and binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs > thr).long()
        masks_long = masks.long()
        
        # 3. Update and log collections
        # Disabled to save memory
        # prob_output = self.train_prob_metrics(probs, masks_long)
        bin_output = self.train_bin_metrics(preds, masks_long)
        # Logging these can be very expensive, especially with large metrics like AUROC that store internal state.
        # self.log_dict(prob_output, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(bin_output, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

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

        # Calculate weighted combined loss
        total_loss = 0.0
        # Ensure masks are floats for BCE and SMP losses
        masks_float = masks.float() 
        
        for name, criterion in self.losses.items():
            loss_val = criterion(logits, masks_float)
            weight = self.loss_weights[name]
            total_loss += weight * loss_val
            
            # Log individual loss components for WandB graphs
            self.log(f"val/loss_{name}", loss_val, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/loss_total", total_loss, on_step=False, on_epoch=True, sync_dist=True)

        # Calculate and log dynamic metrics
        # 1. Get threshold from config (default to 0.5 if not found)
        thr = getattr(self.cfg.evaluation.settings, "threshold", 0.5)
        
        # 2. Get probabilities and binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs > thr).long()
        masks_long = masks.long()
        
        # 3. Update and log collections
        prob_output = self.val_prob_metrics(probs, masks_long)
        bin_output = self.val_bin_metrics(preds, masks_long)
        self.log_dict(prob_output, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(bin_output, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Runs exactly like validation_step, but logs to the 'test/' prefix.
        Used at the very end of training on the held-out test set.
        """
        imgs, masks = batch
        logits = self(imgs)

        total_loss = 0.0
        masks_float = masks.float() 
        
        for name, criterion in self.losses.items():
            loss_val = criterion(logits, masks_float)
            total_loss += self.loss_weights[name] * loss_val
            self.log(f"test/loss_{name}", loss_val, on_step=False, on_epoch=True, sync_dist=True)

        self.log("test/loss_total", total_loss, on_step=False, on_epoch=True, sync_dist=True)

        # 1. Get threshold from config (default to 0.5 if not found)
        thr = getattr(self.cfg.evaluation.settings, "threshold", 0.5)
        
        # 2. Get probabilities and binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs > thr).long()
        masks_long = masks.long()
        
        # 3. Update and log collections
        prob_output = self.test_prob_metrics(probs, masks_long)
        bin_output = self.test_bin_metrics(preds, masks_long)
        self.log_dict(prob_output, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(bin_output, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

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
