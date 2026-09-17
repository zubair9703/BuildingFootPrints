---
name: footprint-model
description: Model and training implementation for Footprint Extraction using satellite-pretrained DINOv3, LoRA, DPT/CNN decoders, PyTorch Lightning, segmentation losses, schedulers, early stopping, and evaluation. Use when modifying model or training code.
---

# Footprint Model

## Architecture

Use:

```text
satellite-pretrained DINOv3
        -> LoRA adaptation
        -> DPT decoder OR lightweight CNN decoder
        -> footprint segmentation logits
```

## DINOv3

Use official Meta DINOv3 implementation and satellite-pretrained weights.

Keep the checkpoint configurable in YAML. Do not hard-code checkpoint paths.

References:

- https://github.com/facebookresearch/dinov3
- https://ai.meta.com/research/dinov3/
- https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

## LoRA

Use LoRA for parameter-efficient backbone fine-tuning.

Example configuration:

```yaml
model:
  lora:
    enabled: true
    rank: 16
    alpha: 32
    dropout: 0.05
    target_modules:
      - qkv
      - proj
```

Default training behavior:

```text
DINOv3 original weights -> frozen
LoRA parameters         -> trainable
decoder                  -> trainable
```

Do not silently unfreeze the full backbone.

Report total parameters, trainable parameters, and trainable percentage.

Reference:
https://arxiv.org/abs/2106.09685

## Decoders

Support two explicit decoder choices selected through YAML:

```yaml
model:
  decoder: dpt
```

or:

```yaml
model:
  decoder: cnn
```

### DPT

Use intermediate DINOv3 features for dense prediction, followed by projection, multi-scale fusion/refinement, and upsampling.

### CNN

Use a lightweight projection + convolution + progressive upsampling decoder.

Do not create a generic decoder framework solely to support these two choices.

## Loss

Use a configurable segmentation loss. A recommended starting point is:

```text
BCEWithLogits + Dice loss
```

Example:

```yaml
loss:
  name: bce_dice
  bce_weight: 0.5
  dice_weight: 0.5
```

Keep loss implementations small and explicit.

## PyTorch Lightning

Use a focused `LightningModule` with:

- `forward()`
- `training_step()`
- `validation_step()`
- `configure_optimizers()`

Use Lightning for training/validation orchestration, mixed precision, gradient accumulation, gradient clipping, checkpointing, logging, device management, and distributed training when required.

Use `EarlyStopping` and `ModelCheckpoint` callbacks.

Use a `LightningDataModule` only when it clearly simplifies dataloader management.

Reference:
https://lightning.ai/docs/pytorch/stable/

## Training

Support at least:

- AdamW
- mixed precision
- gradient accumulation
- gradient clipping
- checkpointing
- resume training
- deterministic seeds

Keep training settings in YAML.

## Early Stopping

Early stopping is mandatory.

Example:

```yaml
training:
  early_stopping:
    enabled: true
    monitor: val_iou
    mode: max
    patience: 15
    min_delta: 0.001
```

Only validation metrics may be used for model selection and early stopping.

## Learning-Rate Scheduling

Support at least:

### Cosine

```yaml
training:
  scheduler:
    name: cosine
    min_lr: 1.0e-6
```

### Reduce on Plateau

```yaml
training:
  scheduler:
    name: reduce_on_plateau
    factor: 0.5
    patience: 5
    min_lr: 1.0e-6
```

Use Lightning's scheduler integration.

## Training-Time Visualisation

Periodically visualise fixed validation samples during training.

Every plot should show:

```text
Image | Ground Truth | Prediction
```

Keep the frequency and sample count configurable, for example:

```yaml
logging:
  plot_every_n_epochs: 5
  num_visualization_samples: 4
```

Run visualisation with `model.eval()` and `torch.no_grad()`. Restore training mode afterward. Visualisation must not participate in gradient computation.

Save plots with experiment outputs and include epoch and tile/sample ID when useful.

A small Lightning callback or validation hook is sufficient. Do not create a visualisation framework.

## Metrics and Evaluation

Report at minimum:

- IoU
- Dice/F1
- precision
- recall

Optionally report pixel accuracy, confusion matrix, per-tile metrics, and per-sample metrics when useful.

Use the test set only for final evaluation after checkpoint selection.

Qualitative evaluation should include difficult cases such as small footprints, dense urban regions, rural buildings, partial occlusion, low contrast, and relevant cloud/shadow conditions.

## Configuration

Keep all experiment parameters in YAML, including:

```text
project
data
augmentation
model
loss
training
evaluation
logging
```

Do not hard-code paths, hyperparameters, checkpoints, decoder selection, LoRA settings, or training controls.
