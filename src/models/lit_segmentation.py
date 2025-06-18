import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex as IoU, BinaryF1Score as Dice



class LitSegmentation(pl.LightningModule):
    """
    PyTorch Lightning wrapper for a segmentation_models.pytorch model.
    Uses BCEWithLogitsLoss + IoU and Dice metrics.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Save hyperparams to self.hparams for logging
        self.save_hyperparameters(cfg)

        # Instantiate the SMP model from config
        self.model: torch.nn.Module = smp.Unet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            classes=cfg.model.classes,
            activation=None  # logits output
        )

        # Loss
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        # Metrics (binary segmentation: 2 classes [bg,fg])

        # self.train_iou = IoU(num_classes=2, ignore_index=0)
        # self.val_iou   = IoU(num_classes=2, ignore_index=0)
        # self.train_dice = Dice(num_classes=2, ignore_index=0)
        # self.val_dice   = Dice(num_classes=2, ignore_index=0)

        self.train_iou = IoU()
        self.val_iou   = IoU()
        self.train_dice = Dice()
        self.val_dice   = Dice()


        # Learning rate
        self.lr = cfg.train.optimizer.lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)

        # Compute metrics on the batch
        preds = (torch.sigmoid(logits) > 0.5).long()
        self.train_iou.update(preds, masks.long())
        self.train_dice.update(preds, masks.long())

        # Log
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/iou",   self.train_iou,   on_step=False, on_epoch=True)
        self.log("train/dice",  self.train_dice,  on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
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

    def configure_optimizers(self):
        # Instantiate optimizer and scheduler from cfg
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.hparams.train.optimizer.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.train.scheduler.T_max,
            eta_min=self.hparams.train.scheduler.eta_min
        )
        return [optimizer], [scheduler]
