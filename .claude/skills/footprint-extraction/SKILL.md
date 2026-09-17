---
name: footprint-extraction
description: Project-wide conventions for the Footprint Extraction research project. Use when implementing, modifying, reviewing, or planning project code that spans data, model, training, evaluation, configuration, or reproducibility.
---

# Footprint Extraction

## Research Code

Keep implementations simple and explicit.

- Write neat, clean, human-readable code.
- Avoid unnecessary classes and functions.
- Avoid factories, registries, wrapper layers, generic managers, and deep inheritance.
- Do not build framework-like abstractions for a research project.
- Keep experiment-specific values in YAML.

## Reproducibility

Every experiment should preserve:

- exact configuration
- seed
- grid/split metadata
- augmentation configuration
- model/LoRA/decoder configuration
- training history
- best and final checkpoints
- test metrics
- qualitative predictions

Never silently change the train/validation/test split.

## Evaluation Principle

The test set is used only for final evaluation after model and checkpoint selection. Early stopping and checkpoint selection must use validation metrics only.

## Related Skills

- `footprint-dataloader` — data preparation and patch sampling.
- `footprint-model` — model, training, and evaluation.
