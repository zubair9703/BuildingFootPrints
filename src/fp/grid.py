"""Spatial grid generation and deterministic train/validation/test splitting.

The complete grid is generated for every source tile *before* any splitting, so
each tile contributes cells to all three splits.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import GriddedPatchSampler, Units

SPLITS = ["train", "val", "test"]


class ImageTile(RasterDataset):
    """RGB imagery: read as float32 and resampled bilinearly."""

    is_image = True


class MaskTile(RasterDataset):
    """Binary footprint mask: read as long and resampled nearest-neighbour."""

    is_image = False


def open_tile(tile_id, config):
    """Open one source tile as an intersection of its imagery and its mask."""
    data = config["data"]
    image_path = Path(data["image_dir"]) / f"{tile_id}.tif"
    mask_path = Path(data["mask_dir"]) / f"{tile_id}.tif"
    res = data["target_res"]

    image = ImageTile(paths=[str(image_path)], res=res)
    mask = MaskTile(paths=[str(mask_path)], res=res)
    return image & mask


def build_grid(config):
    """Generate the complete patch grid for every source tile.

    Returns a GeoDataFrame with one row per grid cell.
    """
    data = config["data"]
    records = []

    for tile_id in data["tiles"]:
        dataset = open_tile(tile_id, config)
        sampler = GriddedPatchSampler(
            dataset,
            size=data["patch_size"],
            stride=data["stride"],
            units=Units.PIXELS,
        )

        cells = list(sampler)

        # Rank coordinates into grid indices: row 0 is the northern-most row.
        col_of = {x: i for i, x in enumerate(sorted({x.start for x, _ in cells}))}
        row_of = {y: i for i, y in enumerate(sorted({y.start for _, y in cells}, reverse=True))}

        for x, y in cells:
            box = shapely.box(x.start, y.start, x.stop, y.stop)
            if not sampler.geometry.contains(box):
                continue
            records.append(
                {
                    "tile_id": tile_id,
                    "grid_row": row_of[y.start],
                    "grid_col": col_of[x.start],
                    "minx": x.start,
                    "miny": y.start,
                    "maxx": x.stop,
                    "maxy": y.stop,
                    "geometry": box,
                }
            )

    grid = gpd.GeoDataFrame(records, geometry="geometry", crs=dataset.crs)
    return grid.sort_values(["tile_id", "grid_row", "grid_col"]).reset_index(drop=True)


def assign_splits(grid, config):
    """Assign each grid cell to train, validation, or test, tile by tile."""
    ratios = config["data"]["split_ratios"]
    rng = np.random.default_rng(config["project"]["seed"])

    grid = grid.copy()
    grid["split"] = ""

    for tile_id in sorted(grid["tile_id"].unique()):
        positions = np.flatnonzero(grid["tile_id"].to_numpy() == tile_id)
        shuffled = rng.permutation(positions)

        n = len(shuffled)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])

        grid.iloc[shuffled[:n_train], grid.columns.get_loc("split")] = "train"
        grid.iloc[shuffled[n_train : n_train + n_val], grid.columns.get_loc("split")] = "val"
        grid.iloc[shuffled[n_train + n_val :], grid.columns.get_loc("split")] = "test"

    return grid


def save_grid(grid, path):
    """Write grid and split metadata to GeoJSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_file(path, driver="GeoJSON")
    return path


def load_grid(path):
    """Read grid and split metadata written by save_grid."""
    return gpd.read_file(path)


def split_counts(grid):
    """Per-tile cell counts for each split, useful for reporting."""
    return grid.pivot_table(
        index="tile_id", columns="split", values="grid_row", aggfunc="count", fill_value=0
    ).reindex(columns=SPLITS, fill_value=0)
