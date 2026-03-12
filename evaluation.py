import argparse
from collections import defaultdict
from typing import Any, Dict, Optional

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
from src.models.factory import build_model, build_tokenizer
from src.models.text_rank import text_rank_summarize
from src.utils.config import load_config
from src.utils.environment import (
    adaptive_display,
    detect_runtime_env,
    try_set_window_position,
)
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metric
from src.utils.utils import (
    find_latest_checkpoint,
    load_checkpoint,
    set_seed,
    token_ids_to_text,
)


class EvaluationEngine:
    """
    Evaluation Engine for comparing multiple summarization models.
    Supports both deep learning models and heuristic baselines (TextRank).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = detect_runtime_env()
        self.logger = setup_logger(name="Evaluator", log_file="evaluation.log")

        # Configurations
        self.models_to_evaluate = [
            "TEXT_RANK",
            "POINTER_GENERATOR_NETWORK",
            "NEURAL_INTRA_ATTENTION_MODEL",
            "TRANSFORMER",
        ]
        self.metrics_list = cfg.validation.metrics
        self.max_tokens = cfg.validation.max_tokens_each_batch

        set_seed(cfg.environment.seed)

    def _setup_dataloader(
        self, model_name: str, tokenizer: Any
    ) -> torch.utils.data.DataLoader:
        """Build specialized dataloader based on model requirements."""
        split = "test"
        if model_name == "TEXT_RANK":
            # TextRank operates on raw text
            ds = CNNDailyMailDataset(split=split, tokenizer=None)
            return torch.utils.data.DataLoader(
                ds, batch_size=4
            )  # Smaller batch for CPU processing

        # DL Models use tokenized input and dynamic batching
        ds = CNNDailyMailDataset(split=split, tokenizer=tokenizer)
        return SummaryDataLoader(
            ds,
            collate_fn=build_collate_fn(tokenizer),
            batch_sampler=DynamicBatchSampler(ds, max_tokens=self.max_tokens),
        )

    def run(self) -> Dict[str, Dict[str, float]]:
        """Execute evaluation loop for all registered models."""
        final_results = defaultdict(dict)

        for model_name in self.models_to_evaluate:
            self.logger.info(f">>> Evaluating model: {model_name}")

            # 1. Initialize Components
            tokenizer = build_tokenizer(model_name)
            loader = self._setup_dataloader(model_name, tokenizer)

            # 2. Load Model Weights (except for TextRank)
            model = None
            if model_name != "TEXT_RANK":
                model = build_model(model_name, tokenizer, self.device, cfg=self.cfg)
                ckpt_path, _ = find_latest_checkpoint(
                    f"{model_name.lower()}_checkpoints"
                )
                if ckpt_path:
                    load_checkpoint(model, ckpt_path, map_location=self.device)
                    self.logger.info(f"Loaded weights from {ckpt_path}")
                else:
                    self.logger.warning(
                        f"No checkpoint found for {model_name}. Using random weights."
                    )

            # 3. Inference & Metric Calculation
            model_metrics = self._evaluate_single_model(
                model_name, model, tokenizer, loader
            )
            final_results[model_name] = model_metrics

        # 4. Visualization
        self._plot_comparison(final_results)
        return final_results

    def _evaluate_single_model(
        self,
        name: str,
        model: Optional[Any],
        tokenizer: Any,
        loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Run inference and aggregate scores for one model."""
        running_scores = defaultdict(list)

        if model:
            model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if name == "TEXT_RANK":
                    candidates = text_rank_summarize(batch["input_text"], 3)
                    references = batch["target_text"]
                else:
                    # Model Inference
                    outputs = model.infer(batch["input_ids"].to(self.device))
                    batch_out_ids = outputs["output_ids"]

                    # Decoding
                    candidates = [
                        token_ids_to_text(tokenizer, ids, oov)
                        for ids, oov in zip(batch_out_ids, batch["oov_list"])
                    ]
                    references = batch["target_text"]

                # Metrics Computation
                batch_results = compute_metric(
                    self.metrics_list, candidates, references
                )
                for m_name, values in batch_results.items():
                    running_scores[m_name].extend(values)

                if batch_idx % 10 == 0:
                    self.logger.info(f"Batch {batch_idx} processed.")

        # Compute Mean
        return {m: sum(v) / len(v) for m, v in running_scores.items()}

    def _plot_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """Generate high-quality bar chart comparison."""
        self.logger.info("Generating comparison plots...")
        x = np.arange(len(self.metrics_list)) * 1.5
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 7))
        for i, model_name in enumerate(self.models_to_evaluate):
            scores = [results[model_name].get(m, 0) for m in self.metrics_list]
            ax.bar(x + i * width, scores, width, label=model_name.replace("_", " "))

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.upper() for m in self.metrics_list])
        ax.set_ylabel("Score")
        ax.set_title(
            "Cross-Architecture Performance Evaluation (CNN/DailyMail Test Set)"
        )
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        try_set_window_position(0, 0)
        adaptive_display(fig, self.env)
        plt.savefig("assets/images/evaluation_comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation Engine")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    engine = EvaluationEngine(config)
    engine.run()
