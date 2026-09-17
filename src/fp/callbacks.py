"""Training-time visualisation of fixed validation samples."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lightning import Callback

from fp.viz import plot_predictions


class PredictionPlotCallback(Callback):
    """Periodically save Image | Ground Truth | Prediction plots for a fixed set
    of validation samples.

    Runs under model.eval() and torch.no_grad(), and restores the training mode
    afterward so visualisation never participates in gradient computation.
    """

    def __init__(self, val_dataset, config, output_dir):
        logging = config["logging"]
        self.every_n_epochs = logging["plot_every_n_epochs"]
        n = min(logging["num_visualization_samples"], len(val_dataset))
        self.indices = list(range(n))
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        was_training = pl_module.training
        pl_module.eval()

        images, masks, predictions = [], [], []
        device = pl_module.device
        with torch.no_grad():
            for index in self.indices:
                sample = self.val_dataset[index]
                image = sample["image"].unsqueeze(0).to(device)
                logits = pl_module(image)
                probability = logits.sigmoid().squeeze(0).squeeze(0)

                images.append(sample["image"])
                masks.append(sample["mask"])
                predictions.append((probability > 0.5).float())

        fig = plot_predictions(
            images, masks, predictions, self.config, title=f"epoch {epoch}"
        )
        fig.savefig(self.output_dir / f"epoch_{epoch:03d}.png", dpi=120)
        plt.close(fig)

        if was_training:
            pl_module.train()
