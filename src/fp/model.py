"""Satellite-pretrained DINOv3 with LoRA adaptation and a DPT or CNN decoder."""

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoBackbone


class ResidualConvUnit(nn.Module):
    """Pre-activation residual block used by the DPT fusion stages."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.norm1(self.conv1(F.relu(x)))
        out = self.norm2(self.conv2(F.relu(out)))
        return x + out


class DPTDecoder(nn.Module):
    """Reassemble four same-scale ViT feature maps to a pyramid, then fuse top-down."""

    def __init__(self, in_channels, channels):
        super().__init__()
        self.project = nn.ModuleList(
            [nn.Conv2d(in_channels, channels, 1) for _ in range(4)]
        )
        # Resample the four stages to 1/4, 1/8, 1/16 and 1/32 of the input.
        self.resample = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels, channels, 4, stride=4),
                nn.ConvTranspose2d(channels, channels, 2, stride=2),
                nn.Identity(),
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            ]
        )
        self.refine = nn.ModuleList([ResidualConvUnit(channels) for _ in range(4)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def forward(self, features):
        pyramid = [
            resample(project(feature))
            for feature, project, resample in zip(features, self.project, self.resample)
        ]

        x = self.refine[3](pyramid[3])
        for i in (2, 1, 0):
            x = F.interpolate(x, size=pyramid[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.refine[i](x + pyramid[i])

        return self.head(x)


class CNNDecoder(nn.Module):
    """Lightweight projection and progressive upsampling of the last feature map."""

    def __init__(self, in_channels, channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, channels, 1)
        self.up1 = self._block(channels, channels // 2)
        self.up2 = self._block(channels // 2, channels // 4)
        self.head = nn.Conv2d(channels // 4, 1, 1)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        x = self.project(features[-1])
        return self.head(self.up2(self.up1(x)))


class FootprintModel(nn.Module):
    """DINOv3 backbone (frozen, LoRA-adapted) followed by a segmentation decoder."""

    def __init__(self, config):
        super().__init__()
        model_config = config["model"]

        backbone = AutoBackbone.from_pretrained(
            model_config["checkpoint"],
            out_indices=model_config["out_indices"],
            reshape_hidden_states=True,
            apply_layernorm=True,
        )
        hidden_size = backbone.config.hidden_size

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        lora = model_config["lora"]
        if lora["enabled"]:
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=lora["rank"],
                    lora_alpha=lora["alpha"],
                    lora_dropout=lora["dropout"],
                    target_modules=list(lora["target_modules"]),
                    bias="none",
                ),
            )
        self.backbone = backbone

        channels = model_config["decoder_channels"]
        if model_config["decoder"] == "dpt":
            self.decoder = DPTDecoder(hidden_size, channels)
        elif model_config["decoder"] == "cnn":
            self.decoder = CNNDecoder(hidden_size, channels)
        else:
            raise ValueError(f"unknown decoder: {model_config['decoder']}")

    def forward(self, image):
        features = self.backbone(pixel_values=image).feature_maps
        logits = self.decoder(features)
        return F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear", align_corners=False
        )


def parameter_summary(model):
    """Total, trainable, and trainable-percentage parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"total parameters:     {total:,}\n"
        f"trainable parameters: {trainable:,}\n"
        f"trainable percentage: {100 * trainable / total:.3f}%\n"
    )


def parameter_groups(model, config):
    """Separate learning rates for the LoRA adapters and the decoder."""
    training = config["training"]
    lora_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": model.decoder.parameters(), "lr": training["decoder_lr"]}]
    if lora_parameters:
        groups.append({"params": lora_parameters, "lr": training["lora_lr"]})
    return groups
