"""Patch datasets and dataloaders built from the saved grid metadata."""

import torch
from torch.utils.data import DataLoader, Dataset

from fp.augment import augment
from fp.grid import open_tile


class PatchDataset(Dataset):
    """Grid cells of one split, read on demand from their source tiles."""

    def __init__(self, grid, config, split, augmented):
        self.cells = grid[grid["split"] == split].reset_index(drop=True)
        self.tiles = {
            tile_id: open_tile(tile_id, config)
            for tile_id in sorted(self.cells["tile_id"].unique())
        }
        self.config = config
        self.augmented = augmented
        self.patch_size = config["data"]["patch_size"]
        self.value_scale = config["data"]["value_scale"]
        self.mean = torch.tensor(config["data"]["mean"]).view(-1, 1, 1)
        self.std = torch.tensor(config["data"]["std"]).view(-1, 1, 1)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        cell = self.cells.iloc[index]
        sample = self.tiles[cell["tile_id"]][
            slice(cell["minx"], cell["maxx"]), slice(cell["miny"], cell["maxy"])
        ]

        image = sample["image"] / self.value_scale
        mask = sample["mask"].float()
        assert image.shape[-2:] == (self.patch_size, self.patch_size), image.shape
        assert mask.shape[-2:] == (self.patch_size, self.patch_size), mask.shape

        if self.augmented:
            image, mask = augment(image, mask, self.config["augmentation"])

        image = (image - self.mean) / self.std
        return {"image": image, "mask": mask}


def build_dataloaders(grid, config):
    """Create the train, validation, and test dataloaders."""
    data = config["data"]
    loaders = {}

    for split in ["train", "val", "test"]:
        is_train = split == "train"
        dataset = PatchDataset(grid, config, split, augmented=is_train)
        loaders[split] = DataLoader(
            dataset,
            batch_size=data["batch_size"],
            shuffle=is_train,
            num_workers=data["num_workers"],
            persistent_workers=data["num_workers"] > 0,
            pin_memory=True,
            drop_last=is_train,
        )

    return loaders
