"""Stage 2: train the footprint segmentation model.

Usage:
    uv run python scripts/train.py --config configs/default.yaml
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from fp.callbacks import PredictionPlotCallback
from fp.config import load_config, make_run_dir, set_seed
from fp.data import PatchDataset, build_dataloaders
from fp.grid import load_grid
from fp.lightning_module import FootprintSegmenter
from fp.model import parameter_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config)
    run_dir = make_run_dir(config)

    grid = load_grid(config["data"]["grid_path"])
    loaders = build_dataloaders(grid, config)
    val_dataset = PatchDataset(grid, config, "val", augmented=False)

    module = FootprintSegmenter(config)
    summary = parameter_summary(module.model)
    print(summary)
    (run_dir / "model_summary.txt").write_text(summary)

    training = config["training"]
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best",
            monitor="val_iou",
            mode="max",
            save_last=True,
        ),
        PredictionPlotCallback(val_dataset, config, run_dir / "plots"),
    ]
    if training["early_stopping"]["enabled"]:
        early_stopping = training["early_stopping"]
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping["monitor"],
                mode=early_stopping["mode"],
                patience=early_stopping["patience"],
                min_delta=early_stopping["min_delta"],
            )
        )

    trainer = L.Trainer(
        max_epochs=training["max_epochs"],
        precision=training["precision"],
        accumulate_grad_batches=training["accumulate_grad_batches"],
        gradient_clip_val=training["gradient_clip_val"],
        limit_train_batches=training.get("limit_train_batches", 1.0),
        limit_val_batches=training.get("limit_val_batches", 1.0),
        callbacks=callbacks,
        logger=CSVLogger(save_dir=run_dir, name="logs", version=""),
        deterministic=False,
    )

    trainer.fit(module, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
