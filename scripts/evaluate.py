"""Stage 3: evaluate a trained checkpoint on the held-out test split.

The test set is used only here, after checkpoint selection on validation metrics.

Usage:
    uv run python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint outputs/dinov3_lora_dpt/checkpoints/best.ckpt
"""

import argparse
import json
from pathlib import Path

import lightning as L
import torch

from fp.config import load_config
from fp.data import PatchDataset, build_dataloaders
from fp.grid import load_grid
from fp.lightning_module import FootprintSegmenter
from fp.viz import plot_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(config["project"]["output_root"]) / config["project"]["name"]

    grid = load_grid(config["data"]["grid_path"])
    loaders = build_dataloaders(grid, config)

    module = FootprintSegmenter.load_from_checkpoint(args.checkpoint, config=config)

    trainer = L.Trainer(logger=False)
    results = trainer.test(module, dataloaders=loaders["test"])[0]

    metrics_path = run_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"test metrics written to {metrics_path}")
    print(json.dumps(results, indent=2))

    test_dataset = PatchDataset(grid, config, "test", augmented=False)
    n_plots = min(config["evaluation"]["num_test_plots"], len(test_dataset))
    threshold = config["evaluation"]["threshold"]

    module.eval()
    images, masks, predictions = [], [], []
    with torch.no_grad():
        for index in range(n_plots):
            sample = test_dataset[index]
            image = sample["image"].unsqueeze(0).to(module.device)
            probability = module(image).sigmoid().squeeze(0).squeeze(0)
            images.append(sample["image"])
            masks.append(sample["mask"])
            predictions.append((probability > threshold).float())

    test_plots_dir = run_dir / "test_plots"
    test_plots_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_predictions(images, masks, predictions, config, title="test set")
    fig.savefig(test_plots_dir / "test_predictions.png", dpi=120)


if __name__ == "__main__":
    main()
