# Footprint Extraction — Claude Code Instructions

## Project

Research-oriented deep learning project for satellite-image footprint extraction.

## Core Requirements

- Use **UV** for Python environment and dependency management.
- Use **TorchGeo `GriddedPatchSampler`** for patch extraction.
- Generate the complete spatial grid for each source tile **before** splitting.
- Split grid cells within each tile into train, validation, and test deterministically.
- Apply geometric transforms identically to imagery and footprint masks.
- Use satellite-pretrained **DINOv3** with **LoRA** fine-tuning.
- Support **DPT** and lightweight **CNN** decoders.
- Use **PyTorch Lightning** for training.
- Use validation metrics for checkpoint selection and early stopping; never use test metrics for selection.
- During training, periodically save `Image | Ground Truth | Prediction` plots from fixed validation samples.
- Keep all experiment parameters in YAML.

## Code Style

Write simple, explicit research code.

- Keep code neat, clean, and human readable.
- Do not add unnecessary classes or functions.
- Avoid factories, registries, wrapper layers, generic managers, and deep abstractions.
- Avoid duplicate logic, hidden state, magic numbers, and hard-coded experiment parameters.
- Prefer the smallest implementation that clearly expresses the research method.

## Repository Structure

Use the following Claude Code skills when relevant:

- `.claude/skills/footprint-extraction/SKILL.md` — project-wide research conventions.
- `.claude/skills/footprint-dataloader/SKILL.md` — geospatial data loading, grids, splits, augmentation, and visualisation.
- `.claude/skills/footprint-model/SKILL.md` — DINOv3, LoRA, decoders, Lightning training, losses, schedulers, and evaluation.

Read the relevant skill before making substantial changes in that area.

## Data-Split Rule

The project intentionally distributes different spatial regions of each source tile across train, validation, and test. This can introduce spatial correlation between splits. Preserve this design unless a spatially blocked experiment is explicitly requested.

Always save the generated grid and split metadata so an experiment can be reproduced exactly.

## Configuration

Do not hard-code experiment-specific values. Put paths, patch size, stride, split ratios, augmentation, model/checkpoint selection, LoRA settings, decoder, loss, optimizer, learning rate, scheduler, batch size, epochs, and early stopping settings in YAML.

## UV

Use:

```bash
uv add <package>
uv remove <package>
uv sync
uv run <command>
```

Declare dependencies in `pyproject.toml` and commit `uv.lock`. Do not maintain a manually edited `requirements.txt` unless an external tool requires it.

For PyTorch, use the current official UV + PyTorch instructions and select the appropriate CPU/CUDA build for the target environment. Do not hard-code an unverified CUDA wheel URL.

## Reproducibility

Preserve, per experiment:

- exact YAML configuration
- random seed
- grid metadata
- split metadata
- model and LoRA configuration
- augmentation configuration
- training history
- best and final checkpoints
- final test metrics
- qualitative prediction plots

## Useful References

- TorchGeo: https://docs.torchgeo.org/
- DINOv3: https://github.com/facebookresearch/dinov3
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- UV: https://docs.astral.sh/uv/
