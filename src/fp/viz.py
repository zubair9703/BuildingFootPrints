"""Matplotlib visualisation of samples, grids, splits, and predictions."""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}


def denormalize(image, config):
    """Undo normalisation so a patch can be displayed as RGB."""
    mean = np.array(config["data"]["mean"]).reshape(-1, 1, 1)
    std = np.array(config["data"]["std"]).reshape(-1, 1, 1)
    image = image.detach().cpu().numpy() * std + mean
    return np.clip(image.transpose(1, 2, 0), 0, 1)


def plot_sample(image, mask, config, title=None):
    """Show one patch as image | ground truth | overlay."""
    rgb = denormalize(image, config)
    mask = mask.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(rgb)
    axes[2].imshow(np.ma.masked_where(mask < 0.5, mask), cmap="autumn", alpha=0.5)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _tile_thumbnail(image_path, max_size=1200):
    """Read a downsampled RGB overview of a source tile plus its extent."""
    with rasterio.open(image_path) as src:
        scale = max(src.width, src.height) / max_size
        width = max(int(src.width / scale), 1)
        height = max(int(src.height / scale), 1)
        data = src.read(out_shape=(src.count, height, width), resampling=Resampling.average)
        bounds = src.bounds
    return data.transpose(1, 2, 0), (bounds.left, bounds.right, bounds.bottom, bounds.top)


def plot_tile_grid(grid, tile_id, image_path, label_every=4):
    """Show the source tile with its complete patch grid drawn on top."""
    cells = grid[grid["tile_id"] == tile_id]
    thumbnail, extent = _tile_thumbnail(image_path)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(thumbnail, extent=extent)
    cells.boundary.plot(ax=ax, color="white", linewidth=0.6)

    for _, cell in cells.iterrows():
        if cell["grid_row"] % label_every == 0 and cell["grid_col"] % label_every == 0:
            ax.text(
                cell["geometry"].centroid.x,
                cell["geometry"].centroid.y,
                f"{cell['grid_row']},{cell['grid_col']}",
                color="white",
                fontsize=6,
                ha="center",
                va="center",
            )

    ax.set_title(f"{tile_id} — {len(cells)} grid cells")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    fig.tight_layout()
    return fig


def plot_split_grid(grid, tile_id, image_path=None):
    """Show train, validation, and test grid cells over the full tile."""
    cells = grid[grid["tile_id"] == tile_id]

    fig, ax = plt.subplots(figsize=(9, 9))
    if image_path is not None:
        thumbnail, extent = _tile_thumbnail(image_path)
        ax.imshow(thumbnail, extent=extent)

    for split, color in SPLIT_COLORS.items():
        subset = cells[cells["split"] == split]
        if not subset.empty:
            subset.plot(ax=ax, color=color, alpha=0.55, edgecolor="white", linewidth=0.6)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.55, label=f"{split} ({(cells['split'] == split).sum()})")
            for split, color in SPLIT_COLORS.items()
        ],
        loc="upper right",
    )

    ax.set_title(f"{tile_id} — split assignment")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    fig.tight_layout()
    return fig


def plot_predictions(images, masks, predictions, config, title=None):
    """Show Image | Ground Truth | Prediction for a batch of samples."""
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)

    for row in range(n):
        rgb = denormalize(images[row], config)
        axes[row][0].imshow(rgb)
        axes[row][1].imshow(masks[row].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row][2].imshow(predictions[row].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        for column, name in enumerate(["Image", "Ground Truth", "Prediction"]):
            axes[row][column].axis("off")
            if row == 0:
                axes[row][column].set_title(name)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
