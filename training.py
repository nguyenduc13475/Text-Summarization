import os
import pickle
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from environment import adaptive_display, detect_runtime_env, try_set_window_position
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

MODEL = "POINTER_GENERATOR_NETWORK"
CHECKPOINT_FOLDER = f"{MODEL.lower()}_checkpoints"
NUM_EPOCHS = 200
MAX_TOKENS_EACH_BATCH = 8000
TRAIN_DATASET_LENGTH = 10000
VALIDATION_DATASET_LENGTH = 1500
CONTINUE_TRAINING = True
LAST_TRAIN_STEP_FILE = f"{CHECKPOINT_FOLDER}/last_train_step.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_LOG_MODE = "graph"
LOSS_LOG_INTERVAL = 10
ENV = detect_runtime_env()
METRICS = ["rouge1", "rouge2", "rougeL", "bleu4", "meteor", "bertscore", "moverscore"]
MODEL_SAVE_INTERVAL = 10
CHECKPOINT_INTERVAL = 10

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
        figure = plt.figure(figsize=(10, 10))
        batch_ax = figure.add_subplot(2, 2, 1)
        epoch_ax = figure.add_subplot(2, 2, 2)
        metric_ax = figure.add_subplot(2, 2, 3)
        batch_ax.set_xlabel("Batch/Epoch Progress")
        batch_ax.set_ylabel("Average Loss Per Token")
        batch_ax.set_title(f"Loss Curves")
        batch_ax.grid(True)
        batch_line_2ds = defaultdict(lambda: None)

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
        adaptive_display(figure, ENV)

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
                shuffle=False,
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
                embedding_dim=512,
                encoder_hidden_dim=512,
                decoder_hidden_dim=512,
                attention_dim=512,
                bottle_neck_dim=512,
                num_layers=6,
                cov_loss_factor=1.0,
                learning_rate=1e-3,
                device=DEVICE,
            )
        case "NEURAL_INTRA_ATTENTION_MODEL":
            model = NeuralIntraAttentionModel(
                tokenizer=tokenizer,
                embedding_dim=512,
                hidden_dim=512,
                num_layers=6,
                rl_loss_factor=0.75,
                learning_rate=1e-3,
                device=DEVICE,
            )
        case "TRANSFORMER":
            model = Transformer(
                tokenizer=tokenizer,
                d_model=512,
                nhead=8,
                num_layers=6,
                learning_rate=1e-3,
                device=DEVICE,
            )

    checkpoint_file, checkpoint_idx = find_latest_checkpoint()

    def return_default_dict_list():
        return defaultdict(list)

    epoch_loss_history = defaultdict(return_default_dict_list)
    batch_loss_history = defaultdict(list)
    metric_history = defaultdict(list)

    if CONTINUE_TRAINING and checkpoint_file is not None:
        load_checkpoint(model, checkpoint_file, map_location=DEVICE)
        print("Model loaded successfully!")
        with open(LAST_TRAIN_STEP_FILE, "rb") as f:
            (
                latest_epoch_idx,
                latest_batch_idx,
                epoch_loss_history,
                batch_loss_history,
                metric_history,
                latest_epoch_num_tokens,
                latest_raw_batch_loss_history,
            ) = pickle.load(f)
    else:
        clear_checkpoint_folder()
        checkpoint_idx = -1
        CONTINUE_TRAINING = False

    save_count = 0
    for epoch in range(NUM_EPOCHS):
        if CONTINUE_TRAINING and epoch < latest_epoch_idx:
            continue
        metrics = defaultdict(list)
        for split in ["train", "validation"]:
            batch_step = (
                model.train_one_batch if split == "train" else model.validate_one_batch
            )
            if CONTINUE_TRAINING:
                epoch_num_tokens = latest_epoch_num_tokens
                raw_batch_loss_history = latest_raw_batch_loss_history
            else:
                epoch_num_tokens = 0
                raw_batch_loss_history = defaultdict(list)
            for batch_idx, batch in enumerate(loader[split]):
                if CONTINUE_TRAINING and batch_idx <= latest_batch_idx:
                    continue
                else:
                    CONTINUE_TRAINING = False
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
                            max_reinforce_length=200,
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
                    batch_output_ids = model.infer(
                        batch["input_ids"],
                        max_output_length=200,
                        beam_width=3,
                    )["output_ids"]

                    output_texts = [
                        token_ids_to_text(
                            tokenizer,
                            output_ids,
                            oov_list,
                        )
                        for output_ids, oov_list in zip(
                            batch_output_ids, batch["oov_list"]
                        )
                    ]

                    for metric, values in compute_metric(
                        METRICS, output_texts, batch["target_text"]
                    ).items():
                        metrics[metric].extend(values)

                if split == "train" and batch_idx % LOSS_LOG_INTERVAL == 0:
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

                if split == "train" and batch_idx % MODEL_SAVE_INTERVAL == 0:
                    save_checkpoint(
                        model, f"{CHECKPOINT_FOLDER}/checkpoint{checkpoint_idx + 1}.pt"
                    )

                    if ENV == "colab" and os.path.exists("/content/drive/MyDrive"):
                        save_checkpoint(model, "/content/drive/MyDrive/temp_model.pt")

                    with open(LAST_TRAIN_STEP_FILE, "wb") as f:
                        pickle.dump(
                            (
                                epoch,
                                batch_idx,
                                epoch_loss_history,
                                batch_loss_history,
                                metric_history,
                                epoch_num_tokens,
                                raw_batch_loss_history,
                            ),
                            f,
                        )

                    save_count += 1
                    print(
                        f"Model saved successfully! (Check point {checkpoint_idx + 1})"
                    )

                    if save_count % CHECKPOINT_INTERVAL == 0:
                        checkpoint_idx += 1

                torch.cuda.empty_cache()

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
