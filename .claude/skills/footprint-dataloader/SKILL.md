---
name: footprint-dataloader
description: Geospatial data pipeline for Footprint Extraction using TorchGeo. Use when working on tile reading, GriddedPatchSampler, spatial grid generation, train/validation/test splits, geometric augmentation, dataset samples, or dataset visualisation.
---

# Footprint Dataloader

## Patch Sampling

Use TorchGeo:

```python
from torchgeo.samplers import GriddedPatchSampler
```

Configure patch size, stride, ROI, and units through YAML.

Do not use `random_split(dataset, ...)` for the project split.

## Grid Before Split

The complete grid must be generated for **every source tile before splitting**.

Workflow:

```text
source tile
    -> complete grid
    -> grid metadata
    -> per-tile split
    -> train / validation / test datasets
```

Every source tile must contribute grid cells to all three splits according to the configured ratios.

Each grid record should retain at least:

```text
tile_id
grid_row
grid_col
bounds
split
```

Use a deterministic seed and save the resulting metadata.

### Spatial Correlation

This split intentionally places different parts of the same tile in different splits. Nearby patches may therefore be spatially correlated. Document this limitation in experiments.

If spatial independence is required, provide a separate configurable blocked/spatial-separation split rather than silently changing the default strategy.

## Geometric Augmentation

For segmentation, every geometric transformation must be applied identically to image and mask.

Support configurable operations such as:

- horizontal flip
- vertical flip
- 90° rotations
- small-angle rotation
- translation
- scale/resize
- crop
- perspective only when scientifically justified

Use appropriate interpolation:

```text
image -> bilinear/bicubic as appropriate
mask  -> nearest-neighbour
```

Never use an interpolation method on discrete masks that creates fractional labels.

Validation and test should normally use deterministic preprocessing without random augmentation.

## Geospatial Integrity

Preserve alignment between:

```text
image
mask
patch extent
```

Do not silently modify CRS or georeferencing metadata.

## Dataset Output

Keep samples minimal. A typical sample is:

```python
{
    "image": image,
    "mask": mask,
}
```

Include `tile_id` or `grid_id` only when needed for analysis, visualisation, or reproducibility.

Do not create elaborate sample wrapper classes without a real need.

## Visualisation

Provide simple Matplotlib/GeoPandas plotting functions for:

```python
plot_sample(...)
plot_tile_grid(...)
plot_split_grid(...)
```

Sample visualisation should show:

```text
image | ground truth | overlay
```

Grid visualisation should show the tile, patch boundaries, and grid indices where useful.

Split visualisation should show train, validation, and test grid cells over the full tile.

Do not build a plotting framework.

## Configuration

Keep patch sampling, split ratios, seed, augmentation, and visualisation settings in YAML.

## Reference

TorchGeo GriddedPatchSampler:
https://docs.torchgeo.org/en/stable/api/samplers/GriddedPatchSampler.html
