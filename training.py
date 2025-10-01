import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from environment import detect_runtime_env, try_set_window_position
from metrics import compute_metric
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from tokenization import PointerGeneratorTokenizer
from transformer import Transformer
from utils import (
    load_checkpoint,
    name_to_latex,
    print_log_file,
    save_checkpoint,
    set_seed,
    token_ids_to_text,
)

set_seed()

MODEL = "NEURAL_INTRA_ATTENTION_MODEL"
CHECKPOINT_FOLDER = f"{MODEL.lower()}_checkpoints"
NUM_EPOCHS = 2
MAX_TOKENS_EACH_BATCH = 3000
TRAIN_DATASET_LENGTH = 15
VALIDATION_DATASET_LENGTH = 10
CONTINUE_TRAINING = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_LOG_MODE = "graph"
LOSS_LOG_INTERVAL = 3
ENV = detect_runtime_env()
METRICS = ["rouge1", "rouge2"]
MODEL_SAVE_INTERVAL = 2
CHECKPOINT_INTERVAL = 3

if ENV in ("colab", "notebook"):
    from IPython.display import clear_output, display


def find_latest_checkpoint():
    if os.path.exists(CHECKPOINT_FOLDER) and any(
        re.match(r"^checkpoint([1-9]\d*)\.pt$", f)
        for f in os.listdir(CHECKPOINT_FOLDER)
    ):
        latest_checkpoint_idx = max(
            (
                int(m.group(1))
                for f in os.listdir(CHECKPOINT_FOLDER)
                if (m := re.match(r"^checkpoint([1-9]\d*)\.pt$", f))
            )
        )

        return (
            f"{CHECKPOINT_FOLDER}/checkpoint{latest_checkpoint_idx}.pt",
            latest_checkpoint_idx,
        )

    return None, -1


def clear_checkpoint_folder():
    if os.path.exists(CHECKPOINT_FOLDER):
        [
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            for p in (
                os.path.join(CHECKPOINT_FOLDER, f)
                for f in os.listdir(CHECKPOINT_FOLDER)
            )
        ]
    else:
        os.makedirs(CHECKPOINT_FOLDER)


if __name__ == "__main__":
    loss_log_file = open("loss_log.txt", "w")

    if LOSS_LOG_MODE == "graph":
        if LOSS_LOG_INTERVAL is not None:
            figure = plt.figure(figsize=(10, 10))
            batch_ax = figure.add_subplot(2, 2, 1)
            epoch_ax = figure.add_subplot(2, 2, 2)
            metric_ax = figure.add_subplot(2, 2, 3)
            batch_ax.set_xlabel("Batch/Epoch Progress")
            batch_ax.set_ylabel("Average Loss Per Token")
            batch_ax.set_title(f"Loss Curves")
            batch_ax.grid(True)
            batch_line_2ds = defaultdict(lambda: None)
        else:
            figure = plt.figure(figsize=(5, 10))
            epoch_ax, metric_ax = figure.subplots(1, 2).flatten()

        try_set_window_position(0, 0)

        epoch_ax.set_xlabel("Epoch")
        epoch_ax.set_ylabel("Average Loss Per Token")
        epoch_ax.set_title(f"Loss Curves")
        epoch_ax.grid(True)

        metric_ax.set_xlabel("Epoch")
        metric_ax.set_ylabel("Average Metric Per Sample")
        metric_ax.set_title(f"Metric Curves")
        metric_ax.grid(True)

        figure.tight_layout(pad=2.0)
        match ENV:
            case "colab" | "notebook":
                display(figure)
            case "gui":
                plt.pause(0.01)

        epoch_line_2ds = defaultdict(lambda: defaultdict(lambda: None))
        metric_line_2ds = defaultdict(lambda: None)

    if MODEL == "TRANSFORMER":
        tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    else:
        tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")

    ds = {
        "train": CNNDailyMailDataset(
            split="train",
            tokenizer=tokenizer,
            dataset_length=TRAIN_DATASET_LENGTH,
        ),
        "validation": CNNDailyMailDataset(
            split="validation",
            tokenizer=tokenizer,
            dataset_length=VALIDATION_DATASET_LENGTH,
        ),
    }
    loader = {
        "train": DataLoader(
            ds["train"],
            collate_fn=collate_fn,
            batch_sampler=DynamicBatchSampler(
                ds["train"],
                max_tokens=MAX_TOKENS_EACH_BATCH,
                shuffle=True,
            ),
            pin_memory=True if DEVICE == "cuda" else False,
        ),
        "validation": DataLoader(
            ds["validation"],
            collate_fn=collate_fn,
            batch_sampler=DynamicBatchSampler(
                ds["validation"],
                max_tokens=MAX_TOKENS_EACH_BATCH,
                shuffle=False,
            ),
            pin_memory=True if DEVICE == "cuda" else False,
        ),
    }

    match MODEL:
        case "POINTER_GENERATOR_NETWORK":
            model = PointerGeneratorNetwork(
                tokenizer=tokenizer,
                embedding_dim=128,
                encoder_hidden_dim=160,
                decoder_hidden_dim=196,
                attention_dim=224,
                bottle_neck_dim=56,
                cov_loss_factor=0.75,
                learning_rate=1e-3,
                device=DEVICE,
            )
        case "NEURAL_INTRA_ATTENTION_MODEL":
            model = NeuralIntraAttentionModel(
                tokenizer=tokenizer,
                embedding_dim=128,
                hidden_dim=160,
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

    checkpoint_file, checkpoint_idx = find_latest_checkpoint()

    if CONTINUE_TRAINING and checkpoint_file is not None:
        load_checkpoint(model, checkpoint_file, map_location=DEVICE)
        print("Model loaded successfully!")
    else:
        clear_checkpoint_folder()
        checkpoint_idx = -1

    epoch_loss_history = defaultdict(lambda: defaultdict(list))
    batch_loss_history = defaultdict(list)
    metric_history = defaultdict(list)
    save_count = 0
    for epoch in range(NUM_EPOCHS):
        metrics = defaultdict(list)
        for split in ["train", "validation"]:
            batch_step = (
                model.train_one_batch if split == "train" else model.validate_one_batch
            )
            epoch_num_tokens = 0
            raw_batch_loss_history = defaultdict(list)
            for batch_idx, batch in enumerate(loader[split]):
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
                    if split == "train":
                        batch_loss_history[loss_type].append(
                            loss_value / batch_num_tokens
                        )
                    raw_batch_loss_history[loss_type].append(loss_value)

                if split == "validation":
                    for input_ids, oov_list, target_text in zip(
                        batch["input_ids"], batch["oov_list"], batch["target_text"]
                    ):
                        output_ids = model.infer(
                            input_ids,
                            max_output_length=5,
                            beam_width=2,
                        )["output_ids"][0]
                        output_text = token_ids_to_text(
                            tokenizer,
                            output_ids,
                            oov_list,
                            tokenizer.get_vocab_size(),
                        )

                        for metric in METRICS:
                            metrics[metric].append(
                                compute_metric(metric, output_text, target_text)
                            )

                if (
                    split == "train"
                    and LOSS_LOG_INTERVAL is not None
                    and batch_idx % LOSS_LOG_INTERVAL == 0
                ):
                    average_loss_per_token = losses["total_loss"] / batch_num_tokens
                    log = f"Epoch {epoch} / Batch {batch_idx} : Average Loss Per Token is {average_loss_per_token}"
                    match LOSS_LOG_MODE:
                        case "console":
                            print(log)
                        case "file":
                            loss_log_file.write(log + "\n")
                        case "graph":
                            for (
                                loss_type,
                                loss_values,
                            ) in batch_loss_history.items():
                                if batch_line_2ds[loss_type] is None:
                                    batch_line_2ds[loss_type] = batch_ax.plot(
                                        loss_values,
                                        label=name_to_latex[loss_type],
                                    )[0]
                                    batch_ax.legend()
                                else:
                                    batch_line_2ds[loss_type].set_xdata(
                                        range(len(loss_values))
                                    )
                                    batch_line_2ds[loss_type].set_ydata(loss_values)

                            batch_ax.relim()
                            batch_ax.autoscale()
                            figure.canvas.draw()
                            figure.canvas.flush_events()
                            if ENV in ("colab", "notebook"):
                                clear_output(wait=True)
                                display(figure)

                if (
                    split == "train"
                    and MODEL_SAVE_INTERVAL is not None
                    and batch_idx % MODEL_SAVE_INTERVAL == 0
                ):
                    save_checkpoint(
                        model, f"{CHECKPOINT_FOLDER}/checkpoint{checkpoint_idx + 1}.pt"
                    )
                    save_count += 1
                    print(
                        f"Model saved successfully! (Check point {checkpoint_idx + 1})"
                    )

                    if (
                        CHECKPOINT_INTERVAL is not None
                        and save_count % CHECKPOINT_INTERVAL == 0
                    ):
                        checkpoint_idx += 1

            for loss_type, loss_values in raw_batch_loss_history.items():
                epoch_loss_history[split][loss_type].append(
                    sum(loss_values) / epoch_num_tokens
                )

            log = f"Epoch {epoch} / {split.capitalize()} : Average Loss Per Token is {epoch_loss_history[split]["total_loss"][-1]}"

            match LOSS_LOG_MODE:
                case "console":
                    print(log)
                case "file":
                    loss_log_file.write(log + "\n")
                case "graph":
                    for (
                        loss_type,
                        loss_values,
                    ) in epoch_loss_history[split].items():
                        if epoch_line_2ds[split][loss_type] is None:
                            epoch_line_2ds[split][loss_type] = epoch_ax.plot(
                                loss_values,
                                label=name_to_latex[loss_type] + f" ({split})",
                            )[0]
                            epoch_ax.legend()
                        else:
                            epoch_line_2ds[split][loss_type].set_xdata(
                                range(len(loss_values))
                            )
                            epoch_line_2ds[split][loss_type].set_ydata(loss_values)

                    epoch_ax.relim()
                    epoch_ax.autoscale()
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    if ENV in ("colab", "notebook"):
                        clear_output(wait=True)
                        display(figure)

        print_log_file(
            "=================================================", loss_log_file
        )
        print_log_file(f"Validation metrics at epoch {epoch} :", loss_log_file)
        for metric, values in metrics.items():
            metrics[metric] = sum(values) / len(values)
            print_log_file(f"{metric.upper()} : {metrics[metric]}", loss_log_file)
            metric_history[metric].append(metrics[metric])
        print_log_file(
            "=================================================", loss_log_file
        )

        if LOSS_LOG_MODE == "graph":
            # tạo 1 graph cho tất cả metrics
            for metric, values in metric_history.items():
                for (
                    metric,
                    values,
                ) in metric_history.items():
                    if metric_line_2ds[metric] is None:
                        metric_line_2ds[metric] = metric_ax.plot(
                            values, label=metric.upper()
                        )[0]
                        metric_ax.legend()
                    else:
                        metric_line_2ds[metric].set_xdata(range(len(values)))
                        metric_line_2ds[metric].set_ydata(values)

                metric_ax.relim()
                metric_ax.autoscale()
                figure.canvas.draw()
                figure.canvas.flush_events()
                if ENV in ("colab", "notebook"):
                    clear_output(wait=True)
                    display(figure)

    loss_log_file.close()
    if LOSS_LOG_MODE == "graph" and ENV == "gui":
        plt.show()
