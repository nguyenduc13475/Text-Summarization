from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from sklearn.manifold import TSNE

from src.utils.environment import adaptive_display
from src.utils.utils import name_to_latex


def plot_attention_heatmap(
    figure: plt.Figure,
    env_mode: str,
    ax: plt.Axes,
    attention_distributions: torch.Tensor,
    output_tokens: List[str],
    input_tokens: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    title_font_size: int = 16,
    axis_font_size: int = 14,
    show_color_bar: bool = True,
    display_immediately: bool = True,
) -> None:
    """Plots an attention heatmap between input and output tokens."""
    im = ax.imshow(
        attention_distributions.detach().cpu().numpy(),
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )

    ax.set_xticks(range(len(input_tokens)))
    ax.set_xticklabels(
        input_tokens, rotation=90, fontsize=min(700 / len(input_tokens), 8)
    )
    ax.set_yticks(range(len(output_tokens)))
    ax.set_yticklabels(output_tokens, fontsize=min(200 / len(output_tokens), 8))
    ax.set_xlabel(xlabel, fontsize=axis_font_size)
    ax.set_ylabel(ylabel, fontsize=axis_font_size)
    ax.set_title(title, fontsize=title_font_size)

    if show_color_bar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention weight", fontsize=axis_font_size)
    if display_immediately:
        adaptive_display(figure, env_mode)


def plot_tsne_embeddings(
    figure: plt.Figure,
    env_mode: str,
    ax: plt.Axes,
    embeddings: torch.Tensor,
    tokens: List[str],
    title: str,
    sample_size: Optional[int] = 300,
) -> None:
    """Plots a T-SNE 2D projection of token embeddings."""
    X = embeddings.detach().cpu().numpy()
    X_unique, idx = np.unique(X, axis=0, return_index=True)
    tokens_unique = [tokens[i] for i in idx]

    if sample_size and len(tokens) > sample_size:
        idx = torch.randperm(len(tokens))[:sample_size].tolist()
        X = X[idx]
        tokens = [tokens[i] for i in idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    X_2d = tsne.fit_transform(X_unique)

    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=20, alpha=0.6)

    for i, token in enumerate(tokens_unique):
        ax.annotate(token, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)

    ax.set_title(title)
    adaptive_display(figure, env_mode)


class LiveTrainingPlotter:
    """
    The class manages the state and updates the graph in real-time during training.
    """

    def __init__(self, env_mode: str) -> None:
        self.env_mode = env_mode
        self.figure = plt.figure(figsize=(10, 5))

        self.batch_ax = self.figure.add_subplot(1, 2, 1)
        self.epoch_ax = self.figure.add_subplot(1, 2, 2)

        self.batch_ax.set_xlabel("Batch/Epoch Progress")
        self.batch_ax.set_ylabel("Average Loss Per Token")
        self.batch_ax.set_title("Batch Loss Curves")
        self.batch_ax.grid(True)

        self.epoch_ax.set_xlabel("Epoch")
        self.epoch_ax.set_ylabel("Average Loss Per Token")
        self.epoch_ax.set_title("Epoch Loss Curves")
        self.epoch_ax.grid(True)

        self.batch_line_2ds = defaultdict(lambda: None)
        self.epoch_line_2ds = defaultdict(lambda: defaultdict(lambda: None))

        from src.utils.environment import try_set_window_position

        try_set_window_position(0, 0)
        self.figure.tight_layout(pad=2.0)
        adaptive_display(self.figure, self.env_mode)

    def update_batch_plot(self, batch_loss_history: Dict[str, List[float]]) -> None:
        for loss_type, loss_values in batch_loss_history.items():
            if self.batch_line_2ds[loss_type] is None:
                self.batch_line_2ds[loss_type] = self.batch_ax.plot(
                    loss_values, label=name_to_latex.get(loss_type, loss_type)
                )[0]
                self.batch_ax.legend()
            else:
                self.batch_line_2ds[loss_type].set_xdata(range(len(loss_values)))
                self.batch_line_2ds[loss_type].set_ydata(loss_values)
        self._draw_and_flush()

    def update_epoch_plot(
        self, epoch_loss_history_split: Dict[str, List[float]], split: str
    ) -> None:
        for loss_type, loss_values in epoch_loss_history_split.items():
            if self.epoch_line_2ds[split][loss_type] is None:
                self.epoch_line_2ds[split][loss_type] = self.epoch_ax.plot(
                    loss_values,
                    label=f"{name_to_latex.get(loss_type, loss_type)} ({split})",
                )[0]
                self.epoch_ax.legend()
            else:
                self.epoch_line_2ds[split][loss_type].set_xdata(range(len(loss_values)))
                self.epoch_line_2ds[split][loss_type].set_ydata(loss_values)
        self._draw_and_flush()

    def _draw_and_flush(self) -> None:
        self.batch_ax.relim()
        self.batch_ax.autoscale()
        self.epoch_ax.relim()
        self.epoch_ax.autoscale()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        if self.env_mode in ("colab", "notebook"):
            clear_output(wait=True)
            display(self.figure)
