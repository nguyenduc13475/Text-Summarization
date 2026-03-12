import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from src.data.dataset import (
    CNNDailyMailDataset,
    DynamicBatchSampler,
    SummaryDataLoader,
    build_collate_fn,
)
from src.models.factory import build_model, build_tokenizer
from src.utils.config import load_config
from src.utils.environment import detect_runtime_env, try_set_window_position
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metric
from src.utils.utils import load_checkpoint, set_seed, token_ids_to_text


class CheckpointValidationEngine:
    """
    Engine for validating multiple model checkpoints over epochs.
    Provides real-time visualization of metric curves.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.env = detect_runtime_env()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Standardize Logger
        self.logger = setup_logger(
            name="CheckpointValidation", log_file="val_checkpoints.log"
        )

        # Hyperparameters
        self.model_name = cfg.model.name
        self.checkpoint_folder = f"{self.model_name.lower()}_checkpoints"
        self.num_epochs = cfg.training.epochs
        self.max_tokens = cfg.validation.max_tokens_each_batch
        self.metrics_to_compute = cfg.validation.metrics

        set_seed(cfg.environment.seed)
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize data pipeline and model architecture."""
        self.logger.info(f"Initializing components for {self.model_name}...")

        self.tokenizer = build_tokenizer(self.model_name)
        self.collate_fn = build_collate_fn(self.tokenizer)

        # Setup Dataset & Loader
        self.ds = CNNDailyMailDataset(split="validation", tokenizer=self.tokenizer)
        self.loader = SummaryDataLoader(
            self.ds,
            collate_fn=self.collate_fn,
            batch_sampler=DynamicBatchSampler(self.ds, max_tokens=self.max_tokens),
        )

        # Build Model
        self.model = build_model(
            self.model_name, self.tokenizer, self.device, cfg=self.cfg
        )

    def _setup_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Initialize live plotting environment."""
        figure, ax = plt.subplots(figsize=(10, 7))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Score")
        ax.set_title(f"Validation Metrics Growth - {self.model_name}")
        ax.grid(True, linestyle="--", alpha=0.7)
        try_set_window_position(0, 0)
        figure.tight_layout(pad=3.0)
        return figure, ax

    def run(self) -> None:
        """Execute validation loop over all epochs."""
        figure, ax = self._setup_plot()
        metric_line_2ds = defaultdict(lambda: None)
        metric_history = defaultdict(list)

        for epoch in range(self.num_epochs):
            ckpt_path = os.path.join(self.checkpoint_folder, f"checkpoint_{epoch}.pt")

            if not os.path.exists(ckpt_path):
                continue

            try:
                load_checkpoint(self.model, ckpt_path, map_location=self.device)
                self.logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint at {ckpt_path}: {e}")
                continue

            epoch_metrics = self._validate_epoch()

            # Print and Log Results
            print(f"\n--- Epoch {epoch} Validation ---")
            for metric, val in epoch_metrics.items():
                metric_history[metric].append(val)
                print(f"{metric.upper()}: {val:.4f}")

            self._update_plot(figure, ax, metric_history, metric_line_2ds)

        self.logger.info("Validation process completed.")
        if self.env == "gui":
            plt.show()

    def _validate_epoch(self) -> Dict[str, float]:
        """Perform validation on the entire loader for one checkpoint."""
        self.model.eval()
        running_metrics = defaultdict(list)
        num_samples = 0

        with torch.no_grad():
            for batch in self.loader:
                num_samples += len(batch["input_ids"])

                # Model Inference
                outputs = self.model.infer(batch["input_ids"])
                batch_output_ids = outputs["output_ids"]

                # Convert IDs to Text (preserving OOV logic)
                output_texts = [
                    token_ids_to_text(self.tokenizer, out_ids, oov)
                    for out_ids, oov in zip(batch_output_ids, batch["oov_list"])
                ]

                # Compute Metrics
                batch_results = compute_metric(
                    self.metrics_to_compute, output_texts, batch["target_text"]
                )

                for metric, values in batch_results.items():
                    running_metrics[metric].extend(values)

                if num_samples % 100 == 0 or num_samples == len(self.ds):
                    self.logger.info(f"Validated {num_samples}/{len(self.ds)} samples")

                torch.cuda.empty_cache()

        # Average metrics for the epoch
        return {m: sum(v) / len(v) for m, v in running_metrics.items()}

    def _update_plot(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        history: Dict[str, List[float]],
        lines: Dict[str, Any],
    ) -> None:
        """Update the live plot with new data points."""
        for metric, values in history.items():
            if lines[metric] is None:
                lines[metric] = ax.plot(values, label=metric.upper(), marker="o")[0]
                ax.legend()
            else:
                lines[metric].set_xdata(range(len(values)))
                lines[metric].set_ydata(values)

        ax.relim()
        ax.autoscale()
        fig.canvas.draw()
        fig.canvas.flush_events()

        if self.env in ("colab", "notebook"):
            from IPython.display import clear_output, display

            clear_output(wait=True)
            display(fig)


if __name__ == "__main__":
    config = load_config()
    engine = CheckpointValidationEngine(config)
    engine.run()
