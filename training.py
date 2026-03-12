import argparse
import os
import pickle
import shutil
from collections import defaultdict
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from src.data.dataset import (
    CNNDailyMailDataset,
    DynamicBatchSampler,
    SummaryDataLoader,
    build_collate_fn,
)
from src.models.factory import ModelArchitecture, build_model, build_tokenizer
from src.utils.config import load_config
from src.utils.environment import detect_runtime_env, get_device_report
from src.utils.hpc_utils import optimize_cuda_performance
from src.utils.logger import setup_logger
from src.utils.paths import get_checkpoint_dir
from src.utils.utils import load_checkpoint, save_checkpoint, set_seed
from src.utils.visualization import LiveTrainingPlotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text Summarization Training Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "TRANSFORMER",
            "POINTER_GENERATOR_NETWORK",
            "NEURAL_INTRA_ATTENTION_MODEL",
        ],
        help="Override model",
    )
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--max_tokens", type=int, help="Override max_tokens")
    parser.add_argument(
        "--no_continue", action="store_true", help="Start training from the beginning."
    )
    return parser.parse_args()


class SummarizationTrainer:
    def __init__(self, cfg: OmegaConf, args: argparse.Namespace) -> None:
        # Hardware Optimization
        optimize_cuda_performance()

        self.cfg = cfg
        raw_name = args.model if args.model else cfg.model.name
        self.architecture_name = ModelArchitecture(raw_name)
        self.total_epochs = args.epochs if args.epochs else cfg.training.epochs
        self.max_tokens = (
            args.max_tokens if args.max_tokens else cfg.model.max_tokens_each_batch
        )
        self.continue_training = (
            False if args.no_continue else cfg.training.continue_training
        )

        self.loss_log_mode = cfg.training.loss_log_mode
        self.loss_log_interval = cfg.training.loss_log_interval
        self.model_save_interval = cfg.training.model_save_interval
        self.save_checkpoint_flag = cfg.training.save_checkpoint

        set_seed(cfg.environment.seed)

        self.checkpoint_folder = str(get_checkpoint_dir(self.architecture_name))
        self.temp_model_file = f"{self.checkpoint_folder}/temp_model.pt"
        self.last_train_step_file = f"{self.checkpoint_folder}/last_train_step.pkl"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = detect_runtime_env()

        self.logger = setup_logger(
            name=f"Train_{self.architecture_name}",
            log_file=f"{self.architecture_name.lower()}_training.log",
        )
        self.logger.info("=" * 40)
        self.logger.info(f"ENVIRONMENT: {get_device_report()}")
        self.logger.info("=" * 40)

        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize Tokenizer, Model and Plotter."""
        self.tokenizer = build_tokenizer(self.architecture_name)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        self.model = build_model(
            self.architecture_name, self.tokenizer, self.device, cfg=config_dict
        )

        self.plotter = None
        if self.loss_log_mode == "graph":
            self.plotter = LiveTrainingPlotter(self.env)

    def _get_loaders(
        self, skip_batches_dict: Optional[Dict[str, int]] = None
    ) -> Dict[str, SummaryDataLoader]:
        """
        Dynamic loader creation.
        PyTorch DataLoader with batch_sampler must NOT have batch_size or shuffle set.
        """
        skip_batches_dict = skip_batches_dict or {"train": 0, "validation": 0}
        collate_fn = build_collate_fn(self.tokenizer)

        # Train Loader
        ds_train = CNNDailyMailDataset(split="train", tokenizer=self.tokenizer)
        sampler_train = DynamicBatchSampler(
            ds_train,
            max_tokens=self.max_tokens,
            batch_nums=self.cfg.training.train_batch_nums,
            start_batch=self.cfg.training.train_start_batch,
        )
        train_loader = SummaryDataLoader(
            ds_train,
            batch_sampler=sampler_train,
            collate_fn=collate_fn,
            skip_batches=skip_batches_dict.get("train", 0),
            pin_memory=True,  # Speed up CPU-to-GPU transfer
        )

        # Val Loader
        ds_val = CNNDailyMailDataset(split="validation", tokenizer=self.tokenizer)
        sampler_val = DynamicBatchSampler(
            ds_val,
            max_tokens=self.max_tokens,
            batch_nums=self.cfg.training.validation_batch_nums,
            start_batch=self.cfg.training.validation_start_batch,
        )
        val_loader = SummaryDataLoader(
            ds_val,
            batch_sampler=sampler_val,
            collate_fn=collate_fn,
            skip_batches=skip_batches_dict.get("validation", 0),
        )

        return {"train": train_loader, "validation": val_loader}

    def _load_training_state(
        self,
    ) -> Tuple[int, int, defaultdict, defaultdict, int, defaultdict, int]:
        """Resume training from pkl state."""

        def return_default_dict_list():
            return defaultdict(list)

        epoch_loss_history = defaultdict(return_default_dict_list)
        batch_loss_history = defaultdict(list)
        state = (0, -1, epoch_loss_history, batch_loss_history, 0, defaultdict(list), 0)

        if self.continue_training and os.path.exists(self.temp_model_file):
            try:
                load_checkpoint(
                    self.model, self.temp_model_file, map_location=self.device
                )
                if os.path.exists(self.last_train_step_file):
                    with open(self.last_train_step_file, "rb") as f:
                        state = pickle.load(f)
                self.logger.info(f"Resuming from Epoch {state[0]}, Batch {state[1]}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load state: {e}. Starting from scratch."
                )
        else:
            if os.path.exists(self.checkpoint_folder):
                shutil.rmtree(self.checkpoint_folder)
            os.makedirs(self.checkpoint_folder)

        return state

    def train(self) -> None:
        (
            latest_epoch_idx,
            latest_batch_idx,
            epoch_loss_history,
            batch_loss_history,
            latest_epoch_num_tokens,
            latest_raw_batch_loss_history,
            latest_num_samples,
        ) = self._load_training_state()

        loss_log_file = open("loss_log.txt", "a" if self.continue_training else "w")

        for epoch in range(self.total_epochs):
            if epoch < latest_epoch_idx:
                continue

            # Initialize loaders for the current epoch
            # Only skip batches if we are in the 'latest_epoch'
            skip_train = (latest_batch_idx + 1) if epoch == latest_epoch_idx else 0
            loaders = self._get_loaders(skip_batches_dict={"train": skip_train})

            for split in ["train", "validation"]:
                loader = loaders[split]
                batch_step = (
                    self.model.train_one_batch
                    if split == "train"
                    else self.model.validate_one_batch
                )

                # Reset or Resume Counters
                if (
                    split == "train"
                    and epoch == latest_epoch_idx
                    and latest_batch_idx != -1
                ):
                    epoch_num_tokens = latest_epoch_num_tokens
                    num_samples = latest_num_samples
                    raw_batch_loss_history = latest_raw_batch_loss_history
                else:
                    epoch_num_tokens = 0
                    num_samples = 0
                    raw_batch_loss_history = defaultdict(list)

                for batch_idx, batch in enumerate(loader):
                    # Adjust batch_idx if skipped
                    actual_batch_idx = batch_idx + (
                        skip_train
                        if split == "train" and epoch == latest_epoch_idx
                        else 0
                    )

                    batch_num_tokens = batch["target_length"].sum().item()
                    epoch_num_tokens += batch_num_tokens
                    num_samples += len(batch["input_ids"])

                    # Model Forward & Backward
                    if self.architecture_name == "NEURAL_INTRA_ATTENTION_MODEL":
                        self.model.rl_loss_factor = 3.0 if epoch > 2 else 0.0

                    # Compute losses (Model accepts BatchData dict keys)
                    losses = batch_step(**batch)

                    for loss_type, loss_value in losses.items():
                        if split == "train":
                            batch_loss_history[loss_type].append(
                                loss_value / batch_num_tokens
                            )
                        raw_batch_loss_history[loss_type].append(loss_value)

                    # Logging & Graphing
                    if (
                        split == "train"
                        and actual_batch_idx % self.loss_log_interval == 0
                    ):
                        avg_loss = losses["total_loss"] / batch_num_tokens
                        msg = f"Epoch {epoch} | Batch {actual_batch_idx} | Samples {num_samples} | Loss: {avg_loss:.4f}"
                        self.logger.info(msg)
                        if self.plotter:
                            self.plotter.update_batch_plot(batch_loss_history)

                    # Periodic Saving
                    if (
                        split == "train"
                        and actual_batch_idx > 0
                        and actual_batch_idx % self.model_save_interval == 0
                    ):
                        save_checkpoint(self.model, self.temp_model_file)
                        with open(self.last_train_step_file, "wb") as f:
                            pickle.dump(
                                (
                                    epoch,
                                    actual_batch_idx,
                                    epoch_loss_history,
                                    batch_loss_history,
                                    epoch_num_tokens,
                                    raw_batch_loss_history,
                                    num_samples,
                                ),
                                f,
                            )

                # End of Split Logic
                for loss_type, loss_values in raw_batch_loss_history.items():
                    epoch_loss_history[split][loss_type].append(
                        sum(loss_values) / epoch_num_tokens
                    )

                if self.plotter:
                    self.plotter.update_epoch_plot(epoch_loss_history[split], split)
                torch.cuda.empty_cache()

            # End of Epoch Saving
            if self.save_checkpoint_flag:
                save_checkpoint(
                    self.model,
                    f"{self.checkpoint_folder}/checkpoint_{epoch}.pt",
                    save_optimizer=False,
                )

        loss_log_file.close()
        if self.loss_log_mode == "graph" and self.env == "gui":
            plt.show()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    trainer = SummarizationTrainer(cfg, args)
    trainer.train()
