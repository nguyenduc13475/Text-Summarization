import logging
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from environment import adaptive_display, detect_runtime_env, try_set_window_position
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from tokenization import PointerGeneratorTokenizer
from transformer import Transformer
from utils import name_to_latex, print_log_file, set_seed

logging.getLogger("datasets").setLevel(logging.ERROR)

MODEL = "POINTER_GENERATOR_NETWORK"
NUM_EPOCHS = 2
MAX_TOKENS_EACH_BATCH = 3000
DATASET_LENGTH = 15
NUM_FOLDS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_LOG_MODE = "console"
LOSS_LOG_INTERVAL = 3
ENV = detect_runtime_env()

if ENV in ("colab", "notebook"):
    from IPython.display import clear_output, display

if __name__ == "__main__":
    loss_log_file = open("loss_log.txt", "w")
    fold_loss_history = []
    cross_validation_losses = []
    if MODEL == "TRANSFORMER":
        tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    else:
        tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")

    for fold in range(NUM_FOLDS):
        set_seed()

        if LOSS_LOG_MODE == "graph":
            plt.close("all")
            if ENV in ("colab", "notebook"):
                clear_output(wait=True)
            n = fold + 1
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

            figure = plt.figure(figsize=(cols * 5, rows * 5))
            try_set_window_position(0, 0)
            axes = figure.subplots(rows, cols)
            if type(axes) == np.ndarray:
                axes = axes.flatten()[:n]
            else:
                axes = np.array([axes])

            for i in range(n):
                if i < fold:
                    for loss_type, loss_values in fold_loss_history[i].items():
                        axes[i].plot(loss_values, label=name_to_latex[loss_type])
                    axes[i].legend()
                axes[i].set_xlabel("Batch/Epoch Progress")
                axes[i].set_ylabel("Average Loss Per Token")
                axes[i].set_title(f"Fold {i} Loss Curves")
                axes[i].grid(True)

            figure.tight_layout(pad=2.0)
            adaptive_display(figure, ENV)

            line_2ds = defaultdict(lambda: None)

        match MODEL:
            case "POINTER_GENERATOR_NETWORK":
                model = PointerGeneratorNetwork(
                    tokenizer=tokenizer,
                    embedding_dim=128,
                    encoder_hidden_dim=160,
                    decoder_hidden_dim=196,
                    attention_dim=224,
                    bottle_neck_dim=56,
                    num_layers=6,
                    cov_loss_factor=0.75,
                    learning_rate=1e-3,
                    device=DEVICE,
                )
            case "NEURAL_INTRA_ATTENTION_MODEL":
                model = NeuralIntraAttentionModel(
                    tokenizer=tokenizer,
                    embedding_dim=128,
                    hidden_dim=160,
                    num_layers=6,
                    rl_loss_factor=0.75,
                    learning_rate=1e-4,
                    device=DEVICE,
                )
            case "TRANSFORMER":
                model = Transformer(
                    tokenizer=tokenizer,
                    d_model=128,
                    nhead=8,
                    num_layers=2,
                    learning_rate=1e-3,
                    device=DEVICE,
                )

        for split in ["train", "cross validation"]:
            ds = CNNDailyMailDataset(
                split=split,
                tokenizer=tokenizer,
                fold=fold,
                num_folds=NUM_FOLDS,
                dataset_length=DATASET_LENGTH,
            )
            loader = DataLoader(
                ds,
                collate_fn=collate_fn,
                batch_sampler=DynamicBatchSampler(
                    ds, max_tokens=MAX_TOKENS_EACH_BATCH, shuffle=True
                ),
                pin_memory=True if DEVICE == "cuda" else False,
            )

            batch_step = (
                model.train_one_batch if split == "train" else model.validate_one_batch
            )
            epoch_loss_history = defaultdict(list)
            for epoch in range(NUM_EPOCHS) if split == "train" else range(1):
                epoch_num_tokens = 0
                batch_loss_history = defaultdict(list)
                for batch_idx, batch in enumerate(loader):
                    batch_num_tokens = batch["target_length"].sum().item()
                    epoch_num_tokens += batch_num_tokens
                    match MODEL:
                        case "POINTER_GENERATOR_NETWORK":
                            losses = batch_step(
                                batch["input_ids"],
                                batch["target_ids"],
                                batch["oov_list"],
                                batch["input_length"],
                            )
                        case "NEURAL_INTRA_ATTENTION_MODEL":
                            losses = batch_step(
                                batch["input_ids"],
                                batch["target_ids"],
                                batch["oov_list"],
                                batch["input_length"],
                                max_reinforce_length=100,
                                target_texts=batch["target_text"],
                            )
                        case "TRANSFORMER":
                            losses = batch_step(
                                batch["input_ids"],
                                batch["target_ids"],
                            )

                    for loss_type, loss_value in losses.items():
                        batch_loss_history[loss_type].append(loss_value)
                        epoch_loss_history[loss_type].append(
                            loss_value / batch_num_tokens
                        )

                    if (
                        split == "train"
                        and LOSS_LOG_INTERVAL is not None
                        and batch_idx % LOSS_LOG_INTERVAL == 0
                    ):
                        average_loss_per_token = losses["total_loss"] / batch_num_tokens
                        log = f"Fold {fold} / Epoch {epoch} / Batch {batch_idx} : Average Loss Per Token is {average_loss_per_token}"
                        match LOSS_LOG_MODE:
                            case "console":
                                print(log)
                            case "file":
                                loss_log_file.write(log + "\n")
                            case "graph":
                                for (
                                    loss_type,
                                    loss_values,
                                ) in epoch_loss_history.items():
                                    if line_2ds[loss_type] is None:
                                        line_2ds[loss_type] = axes[-1].plot(
                                            loss_values, label=name_to_latex[loss_type]
                                        )[0]
                                        axes[-1].legend()
                                    else:
                                        line_2ds[loss_type].set_xdata(
                                            range(len(loss_values))
                                        )
                                        line_2ds[loss_type].set_ydata(loss_values)

                                axes[-1].relim()
                                axes[-1].autoscale()
                                figure.canvas.draw()
                                figure.canvas.flush_events()
                                if ENV in ("colab", "notebook"):
                                    clear_output(wait=True)
                                    display(figure)

                average_loss_per_token = (
                    sum(batch_loss_history["total_loss"]) / epoch_num_tokens
                )

                if split == "train":
                    log = f"Fold {fold} / Epoch {epoch} : Average Loss Per Token is {average_loss_per_token}"
                    match LOSS_LOG_MODE:
                        case "console":
                            print(log)
                        case "file":
                            loss_log_file.write(log + "\n")

            if split == "cross validation":
                cross_validation_losses.append(average_loss_per_token)
            else:
                fold_loss_history.append(epoch_loss_history)

        print_log_file(f"Fold {fold} Loss is {average_loss_per_token}", loss_log_file)

    print_log_file(
        f"Average Fold Loss is {sum(cross_validation_losses) / NUM_FOLDS}",
        loss_log_file,
    )
    loss_log_file.close()
    if LOSS_LOG_MODE == "graph" and ENV == "gui":
        plt.show()
