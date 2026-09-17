# Footprint Extraction

Building-footprint segmentation from 0.06 m RGB satellite tiles, using satellite-pretrained
DINOv3 (SAT-493M) with LoRA fine-tuning and a DPT or lightweight CNN decoder, trained with
PyTorch Lightning.

## Data

```
FP_data/Images_v1/image{1,2,3}.tif   # 3-band uint8 RGB, 0.06 m/px, EPSG:32644
FP_data/Masks_v1/image{1,2,3}.tif    # 1-band uint8 {0,1} building mask, same grid as imagery
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
# CUDA machine (this project targets torch/cu126)
uv sync --extra cu126 --group notebook

# CPU-only machine
uv sync --extra cpu --group notebook
```

`cpu` and `cu126` are conflicting extras — pick exactly one. Verify the install:

```bash
uv run --extra cu126 python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **Repeat `--extra cu126` (or `--extra cpu`) on every `uv run` / `uv sync` call.**
> uv has no "sticky" extra: `torch` is also a plain transitive dependency of `torchgeo` and
> `lightning`, so any bare `uv run ...` without the flag re-resolves the environment and
> **silently reinstalls a CPU-only `torch` wheel from PyPI**, even after a correct
> `--extra cu126` sync. This is a genuine uv footgun, confirmed in this project — always
> include the extra, as every command below does.

### Gated DINOv3 weights

`facebook/dinov3-vitl16-pretrain-sat493m` is a gated Hugging Face repo. Request access on
the [model page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m), then:

```bash
uv run --extra cu126 hf auth login
uv run --extra cu126 hf auth whoami   # confirm you're logged in before training
```

## Pipeline

Three stages, each reading its parameters from a YAML config
(`configs/default.yaml` for the full experiment, `configs/smoke.yaml` for a fast run).
Substitute `--extra cpu` throughout if you synced for CPU.

```bash
# 1. Build the complete spatial grid per tile, then split deterministically
uv run --extra cu126 python scripts/build_grid.py --config configs/default.yaml

# 2. Train (LoRA-adapted DINOv3 backbone + DPT/CNN decoder)
uv run --extra cu126 python scripts/train.py --config configs/default.yaml

# 3. Evaluate the selected checkpoint on the held-out test split
uv run --extra cu126 python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint outputs/dinov3_lora_dpt/checkpoints/best.ckpt
```

See `notebooks/end_to_end.ipynb` for all three stages run together with visualisation, on
the fast `configs/smoke.yaml`. Launch it the same way:

```bash
uv run --extra cu126 jupyter lab notebooks/end_to_end.ipynb
```

## Design notes

- **Grid before split**: `scripts/build_grid.py` generates the complete TorchGeo
  `GriddedPatchSampler` grid for every tile first, then splits cells of each tile
  independently into train/val/test (`src/fp/grid.py`). Every tile therefore contributes to
  all three splits — this is intentional (see `CLAUDE.md`), not a bug. It means nearby
  patches across splits can be spatially correlated; treat validation/test performance with
  that caveat. Grid and split metadata are saved to `data.grid_path` for exact reproduction.
- **Edge cells**: with `patch_size == stride`, `GriddedPatchSampler` centres the grid over
  each tile; on tiles whose extent isn't an exact multiple of the patch size, the outermost
  ring of grid cells would read slightly outside the tile. `build_grid` drops those, keeping
  only cells fully contained in the tile (289 of 361 candidate cells per tile at the default
  512 px / 512 px setting).
- **Alignment**: image and mask tiles share identical extent, resolution, and CRS, so
  `IntersectionDataset` (`image & mask` in `src/fp/grid.py`) needs no reprojection.
  `RasterDataset` resamples imagery bilinearly and masks nearest-neighbour automatically
  (based on dtype), so augmentation (`src/fp/augment.py`) is the only place both are resampled
  again, and it uses the same nearest-neighbour rule for the mask explicitly.
- **Normalisation**: uses DINOv3 SAT-493M statistics (`mean=[0.430,0.411,0.296]`,
  `std=[0.213,0.156,0.143]`), not ImageNet's — set in `configs/*.yaml` under `data`.
- **LoRA target modules**: the Hugging Face DINOv3 implementation names attention
  projections `q_proj`/`k_proj`/`v_proj`/`o_proj` (not the fused `qkv`/`proj` naming used by
  Meta's/timm's implementation) — `configs/*.yaml` targets the HF names.
- **`uv run` and the `cu126` extra**: pass `--extra cu126` on every invocation (see Setup) —
  uv re-resolves the environment on each `uv run`, and without the flag `torch` falls back to
  a transitive CPU wheel pulled in by `torchgeo`/`lightning`, even after a correct sync.
- **Reproducibility bundle** per run, under `outputs/<project.name>/`:
  `config.yaml`, `grid.geojson`, `model_summary.txt`, `checkpoints/{best,last}.ckpt`,
  `logs/metrics.csv`, `plots/epoch_*.png`, and (after evaluation) `test_metrics.json` and
  `test_plots/`.

## Repository layout

```
configs/            default.yaml (full run), smoke.yaml (fast notebook run)
src/fp/
  config.py          load_config, set_seed, make_run_dir
  grid.py            grid generation + deterministic per-tile split
  data.py            patch dataset + dataloaders
  augment.py          paired geometric augmentation
  model.py            DINOv3 + LoRA + DPT/CNN decoder
  losses.py            BCE + Dice loss
  lightning_module.py  FootprintSegmenter (LightningModule)
  callbacks.py          fixed-sample prediction plotting
  viz.py                 sample / grid / split / prediction plots
scripts/             build_grid.py, train.py, evaluate.py
notebooks/           end_to_end.ipynb
```
