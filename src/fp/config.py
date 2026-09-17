"""Configuration loading, seeding, and run directories."""

from pathlib import Path

import lightning as L
import yaml


def load_config(path):
    """Read a YAML experiment configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(config):
    """Seed Python, NumPy, and PyTorch, including dataloader workers."""
    L.seed_everything(config["project"]["seed"], workers=True)


def make_run_dir(config):
    """Create the output directory for this experiment and save the exact config."""
    run_dir = Path(config["project"]["output_root"]) / config["project"]["name"]
    for name in ["checkpoints", "logs", "plots", "test_plots"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return run_dir
