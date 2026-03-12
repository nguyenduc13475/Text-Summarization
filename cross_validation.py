import argparse
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.dataset import (
    CNNDailyMailDataset,
    DynamicBatchSampler,
    SummaryDataLoader,
    build_collate_fn,
)
from src.models.factory import ModelArchitecture, build_model, build_tokenizer
from src.utils.config import load_config
from src.utils.environment import (
    adaptive_display,
    detect_runtime_env,
    try_set_window_position,
)
from src.utils.logger import setup_logger
from src.utils.utils import name_to_latex, set_seed

# Detect environment for plotting
ENV = detect_runtime_env()
if ENV in ("colab", "notebook"):
    from IPython.display import clear_output


class CrossValidationEngine:
    """
    Cross-Validation Engine for Text Summarization.
    Maintains consistency with the training pipeline while supporting K-Fold logic.
    """

    def __init__(self, cfg: DictConfig, model_name: str) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture_name = ModelArchitecture(model_name)

        # Setup Logger
        self.logger = setup_logger(
            name=f"CV_{self.architecture_name}",
            log_file=f"cv_{self.architecture_name.lower()}.log",
        )

        # Hyperparameters from config
        self.num_folds = cfg.cross_validation.num_folds
        self.epochs = cfg.cross_validation.epochs
        self.max_tokens = cfg.cross_validation.max_tokens_each_batch
        self.loss_log_interval = cfg.training.loss_log_interval
        self.loss_log_mode = cfg.training.loss_log_mode

        # Initialize components
        self.tokenizer = build_tokenizer(self.architecture_name.value)
        self.collate_fn = build_collate_fn(self.tokenizer)

        self.fold_loss_history = []
        self.cross_validation_losses = []

    def _setup_plot(
        self, current_fold: int
    ) -> Tuple[Optional[plt.Figure], Optional[np.ndarray], Optional[defaultdict]]:
        """Initializes multi-subplot visualization for folds."""
        if self.loss_log_mode != "graph":
            return None, None, None

        plt.close("all")
        if ENV in ("colab", "notebook"):
            clear_output(wait=True)

        n = current_fold + 1
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        fig = plt.figure(figsize=(cols * 5, rows * 5))
        try_set_window_position(0, 0)
        axes = fig.subplots(rows, cols)

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()[:n]
        else:
            axes = np.array([axes])

        return fig, axes, defaultdict(lambda: None)

    def run_fold(self, fold_idx: int) -> float:
        """Executes training and validation for a single fold."""
        set_seed(self.cfg.environment.seed)
        self.logger.info(f"Starting Fold {fold_idx}/{self.num_folds}...")

        # Visual state for current fold
        fig, axes, line_2ds = self._setup_plot(fold_idx)

        # Build model with specific config
        model = build_model(
            self.architecture_name, self.tokenizer, self.device, cfg=self.cfg
        )

        epoch_loss_history = defaultdict(list)
        average_val_loss = 0.0

        for split in ["train", "cross validation"]:
            # Standardize dataset loading using internal fold logic
            ds = CNNDailyMailDataset(
                split=split,
                tokenizer=self.tokenizer,
                fold=fold_idx,
                num_folds=self.num_folds,
            )
            loader = SummaryDataLoader(
                ds,
                collate_fn=self.collate_fn,
                batch_sampler=DynamicBatchSampler(ds, max_tokens=self.max_tokens),
            )

            # Unified interface: train_one_batch or validate_one_batch
            batch_step = (
                model.train_one_batch if split == "train" else model.validate_one_batch
            )

            # Cross validation usually runs 1 epoch for evaluation
            num_epochs = self.epochs if split == "train" else 1

            for epoch in range(num_epochs):
                total_tokens = 0
                batch_losses = defaultdict(list)

                for batch_idx, batch in enumerate(loader):
                    num_tokens = batch["target_length"].sum().item()
                    total_tokens += num_tokens

                    # Compute losses using kwargs unpacking
                    losses = batch_step(**batch)

                    for l_type, l_val in losses.items():
                        batch_losses[l_type].append(l_val)
                        if split == "train":
                            epoch_loss_history[l_type].append(l_val / num_tokens)

                    # Graphing logic
                    if split == "train" and batch_idx % self.loss_log_interval == 0:
                        if self.loss_log_mode == "graph":
                            self._update_graph(axes, line_2ds, epoch_loss_history, fig)

                # Calculate final loss for this split
                epoch_avg_loss = sum(batch_losses["total_loss"]) / total_tokens
                if split == "cross validation":
                    average_val_loss = epoch_avg_loss
                else:
                    self.fold_loss_history.append(epoch_loss_history)

        self.logger.info(f"Fold {fold_idx} Validation Loss: {average_val_loss:.4f}")
        return average_val_loss

    def _update_graph(
        self,
        axes: np.ndarray,
        line_2ds: Dict[str, Any],
        history: Dict[str, List[float]],
        fig: plt.Figure,
    ) -> None:
        """Updates the live loss curve for the active fold."""
        for l_type, l_values in history.items():
            latex_label = name_to_latex.get(l_type, l_type)
            if line_2ds[l_type] is None:
                line_2ds[l_type] = axes[-1].plot(l_values, label=latex_label)[0]
                axes[-1].legend()
                axes[-1].grid(True)
            else:
                line_2ds[l_type].set_xdata(range(len(l_values)))
                line_2ds[l_type].set_ydata(l_values)

        axes[-1].relim()
        axes[-1].autoscale()
        fig.canvas.draw()
        fig.canvas.flush_events()
        adaptive_display(fig, ENV)

    def execute(self) -> None:
        """Starts the full K-Fold Cross Validation pipeline."""
        for fold in range(self.num_folds):
            loss = self.run_fold(fold)
            self.cross_validation_losses.append(loss)

        avg_cv_loss = sum(self.cross_validation_losses) / self.num_folds
        self.logger.info(f"Final Cross-Validation Score: {avg_cv_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, help="Override model name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Run the engine
    engine = CrossValidationEngine(
        config, args.model if args.model else config.model.name
    )
    engine.execute()
