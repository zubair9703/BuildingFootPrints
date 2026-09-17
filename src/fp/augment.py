"""Geometric augmentation applied identically to imagery and footprint masks."""

import random

import torch
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional import InterpolationMode


def augment(image, mask, config):
    """Apply random geometric transforms to an (image, mask) pair.

    The mask is always resampled with nearest-neighbour interpolation so it keeps
    discrete labels. `image` is (C, H, W) and `mask` is (H, W).
    """
    mask = mask.unsqueeze(0)

    if random.random() < config["horizontal_flip"]:
        image = torch.flip(image, dims=[-1])
        mask = torch.flip(mask, dims=[-1])

    if random.random() < config["vertical_flip"]:
        image = torch.flip(image, dims=[-2])
        mask = torch.flip(mask, dims=[-2])

    if random.random() < config["rot90"]:
        k = random.randint(1, 3)
        image = torch.rot90(image, k, dims=[-2, -1])
        mask = torch.rot90(mask, k, dims=[-2, -1])

    if random.random() < config["affine_probability"]:
        angle = random.uniform(-config["rotation_degrees"], config["rotation_degrees"])
        max_shift = config["translate_fraction"] * image.shape[-1]
        translate = [
            round(random.uniform(-max_shift, max_shift)),
            round(random.uniform(-max_shift, max_shift)),
        ]
        scale = random.uniform(*config["scale_range"])

        image = F.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = F.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
        )

    return image.contiguous(), mask.squeeze(0).contiguous()
