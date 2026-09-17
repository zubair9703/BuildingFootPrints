"""Stage 1: build the complete per-tile grid and its train/val/test split.

Usage:
    uv run python scripts/build_grid.py --config configs/default.yaml
"""

import argparse

from fp.config import load_config, set_seed
from fp.grid import assign_splits, build_grid, save_grid, split_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config)

    grid = build_grid(config)
    grid = assign_splits(grid, config)

    path = save_grid(grid, config["data"]["grid_path"])
    print(f"saved {len(grid)} grid cells across {grid['tile_id'].nunique()} tiles to {path}")
    print(split_counts(grid))


if __name__ == "__main__":
    main()
