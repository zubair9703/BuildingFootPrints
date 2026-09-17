"""Segmentation loss."""

import torch.nn.functional as F


def bce_dice_loss(logits, target, config):
    """Weighted sum of binary cross-entropy and soft Dice loss."""
    bce = F.binary_cross_entropy_with_logits(logits, target)

    probability = logits.sigmoid()
    smooth = config["dice_smooth"]
    intersection = (probability * target).sum(dim=(1, 2, 3))
    union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + smooth) / (union + smooth)).mean()

    return config["bce_weight"] * bce + config["dice_weight"] * dice
