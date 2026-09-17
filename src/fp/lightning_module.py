"""PyTorch Lightning module for footprint segmentation."""

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from fp.losses import bce_dice_loss
from fp.model import FootprintModel, parameter_groups


def _metric_collection():
    return torch.nn.ModuleDict(
        {
            "iou": BinaryJaccardIndex(),
            "f1": BinaryF1Score(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )


class FootprintSegmenter(L.LightningModule):
    """DINOv3 + LoRA + decoder, trained and evaluated with Lightning."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = FootprintModel(config)

        self.val_metrics = _metric_collection()
        self.test_metrics = _metric_collection()

    def forward(self, image):
        return self.model(image)

    def _step(self, batch):
        logits = self(batch["image"])
        target = batch["mask"].unsqueeze(1)
        loss = bce_dice_loss(logits, target, self.config["loss"])
        return loss, logits.sigmoid(), target

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probability, target = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        target = target.long()
        for name, metric in self.val_metrics.items():
            metric.update(probability, target)
            self.log(f"val_{name}", metric, on_epoch=True, prog_bar=(name == "iou"))
        return loss

    def test_step(self, batch, batch_idx):
        loss, probability, target = self._step(batch)
        self.log("test_loss", loss, on_epoch=True)
        target = target.long()
        for name, metric in self.test_metrics.items():
            metric.update(probability, target)
            self.log(f"test_{name}", metric, on_epoch=True)
        return loss

    def configure_optimizers(self):
        training = self.config["training"]
        optimizer = AdamW(
            parameter_groups(self.model, self.config), weight_decay=training["weight_decay"]
        )

        scheduler_config = training["scheduler"]
        if scheduler_config["name"] == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=training["max_epochs"], eta_min=scheduler_config["min_lr"]
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif scheduler_config["name"] == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=scheduler_config["factor"],
                patience=scheduler_config["patience"],
                min_lr=scheduler_config["min_lr"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_iou"},
            }
        else:
            raise ValueError(f"unknown scheduler: {scheduler_config['name']}")
