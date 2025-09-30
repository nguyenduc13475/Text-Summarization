import os
import re
import shutil

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from metrics import compute_metric
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from transformer import Transformer
from utils import set_seed, token_ids_to_text

set_seed()

MODEL = "TRANSFORMER"
CHECKPOINT_FOLDER = f"{MODEL.lower()}_checkpoints"
NUM_EPOCHS = 1000
MAX_TOKENS_EACH_BATCH = 10000
CONTINUE_TRAINING = False
DEVICE = "cpu"
LOSS_LOG_MODE = "console"
LOSS_LOG_INTERVAL = 10
METRICS = ["rouge1", "rouge2", "rougeL", "bleu", "meteor", "bertscore", "moverscore"]
MODEL_SAVE_INTERVAL = 10
CHECKPOINT_INTERVAL = 10  # 10 lần save thì mới checkpoint


def find_latest_checkpoint():
    if os.path.exists(CHECKPOINT_FOLDER) and any(
        re.match(r"^checkpoint([1-9]\d*)\.pt$", f)
        for f in os.listdir(CHECKPOINT_FOLDER)
    ):
        latest_checkpoint = max(
            (
                int(m.group(1))
                for f in os.listdir(CHECKPOINT_FOLDER)
                if (m := re.match(r"^checkpoint([1-9]\d*)\.pt$", f))
            )
        )

        return (
            f"{CHECKPOINT_FOLDER}/checkpoint{latest_checkpoint}.pt",
            latest_checkpoint,
        )

    return None, None


def clear_checkpoint_folder():
    [
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        for p in (
            os.path.join(CHECKPOINT_FOLDER, f) for f in os.listdir(CHECKPOINT_FOLDER)
        )
    ]


if __name__ == "__main__":
    loss_log_file = open("loss_log.txt", "w")

    if LOSS_LOG_MODE == "graph":
        plt.ion()
        if LOSS_LOG_INTERVAL is not None:
            fig, (batch_ax, epoch_ax, metric_ax, _) = plt.subplots(
                2, 2, figsize=(10, 6)
            )
        else:
            fig, (epoch_ax, metric_ax) = plt.subplots(1, 2, figsize=(10, 6))

    ds = {
        "train": CNNDailyMailDataset(),
        "validation": CNNDailyMailDataset(split="validation"),
    }
    loader = {
        "train": DataLoader(
            ds["train"],
            collate_fn=collate_fn,
            batch_sampler=DynamicBatchSampler(
                ds["train"], max_tokens=MAX_TOKENS_EACH_BATCH, shuffle=True
            ),
        ),
        "validation": DataLoader(
            ds["validation"],
            collate_fn=collate_fn,
            batch_sampler=DynamicBatchSampler(
                ds["validation"], max_tokens=MAX_TOKENS_EACH_BATCH, shuffle=True
            ),
        ),
    }

    checkpoint_file, checkpoint_idx = find_latest_checkpoint()

    if CONTINUE_TRAINING and checkpoint_file is not None:
        model = torch.load(checkpoint_file)
    else:
        clear_checkpoint_folder()

        match MODEL:
            case "POINTER_GENERATOR_NETWORK":
                model = PointerGeneratorNetwork(
                    vocab_size=ds["train"].vocab_size,
                    tokenizer=ds["train"].tokenizer,
                    embedding_dim=128,
                    encoder_hidden_dim=160,
                    decoder_hidden_dim=196,
                    attention_dim=224,
                    bottle_neck_dim=56,
                    cov_loss_factor=0.75,
                    learning_rate=1e-3,
                )
            case "NEURAL_INTRA_ATTENTION_MODEL":
                model = NeuralIntraAttentionModel(
                    vocab_size=ds["train"].vocab_size,
                    tokenizer=ds["train"].tokenizer,
                    embedding_dim=128,
                    hidden_dim=160,
                    unknown_token=ds["train"].tokenizer.token_to_id("<unk>"),
                    start_token=ds["train"].tokenizer.token_to_id("<s>"),
                    end_token=ds["train"].tokenizer.token_to_id("</s>"),
                )
            case "TRANSFORMER":
                model = Transformer(
                    vocab_size=ds["train"].vocab_size,
                    tokenizer=ds["train"].tokenizer,
                    d_model=128,
                    nhead=8,
                    num_layers=2,
                    learning_rate=1e-3,
                )

    epoch_loss_history = dict()
    metric_history = dict()
    for metric in METRICS:
        metric_history[metric] = []
    save_count = 0
    for epoch in range(NUM_EPOCHS):
        metrics = dict()
        for metric in METRICS:
            metrics[metric] = []
        for split in ["train", "validation"]:
            batch_loss_history = None
            caller = (
                model.train_one_batch if split == "train" else model.validate_one_batch
            )
            for batch_idx, batch in enumerate(loader[split]):
                match MODEL:
                    case "POINTER_GENERATOR_NETWORK":
                        losses = caller(
                            batch["input_ids"], batch["labels"], batch["input_lengths"]
                        )
                    case "NEURAL_INTRA_ATTENTION_MODEL":
                        losses = caller(
                            batch["input_ids"],
                            batch["labels"],
                            batch["oov_lists"],
                            batch["input_lengths"],
                        )
                    case "TRANSFORMER":
                        losses = caller(
                            batch["input_ids"], batch["labels"], batch["input_lengths"]
                        )

                if batch_loss_history is None:
                    batch_loss_history = {k: [] for k in losses.keys()}

                for k, v in losses.items():
                    batch_loss_history[k].append(v.item())

                if split == "validation":
                    for sample_idx in len(batch["input_ids"]):
                        output_ids = model.infer(batch["input_ids"][sample_idx])
                        candidate_summary = token_ids_to_text(
                            ds["validation"].tokenizer,
                            output_ids,
                            batch["oov_lists"][sample_idx],
                            ds["validation"].vocab_size,
                        )
                        reference_summary = token_ids_to_text(
                            ds["validation"].tokenizer,
                            batch["labels"][sample_idx],
                            batch["oov_lists"][sample_idx],
                            ds["validation"].vocab_size,
                        )

                        for metric in METRICS:
                            metrics[metric].append(
                                compute_metric(
                                    metric, candidate_summary, reference_summary
                                )
                            )

                if (
                    split == "train"
                    and LOSS_LOG_INTERVAL is not None
                    and batch_idx % LOSS_LOG_INTERVAL == 0
                ):
                    match LOSS_LOG_MODE:
                        case "console":
                            print(
                                f"Epoch {epoch} / Batch {batch_idx} : Loss {losses["total_loss"]}"
                            )
                        case "file":
                            loss_log_file.write(
                                f"Epoch {epoch} / Batch {batch_idx} : Loss {losses["total_loss"]}\n"
                            )
                        case "graph":
                            for k, v_list in batch_loss_history.items():
                                batch_ax.plot(v_list, label=k)
                            batch_ax.set_xlabel("Batch (Epoch progress)")
                            batch_ax.set_ylabel("Loss")
                            batch_ax.set_title(f"Loss Curves")
                            batch_ax.legend()
                            batch_ax.grid(True)
                            plt.pause(0.01)
                            batch_ax.cla()

                if (
                    split == "train"
                    and MODEL_SAVE_INTERVAL is not None
                    and batch_idx % LOSS_LOG_INTERVAL == 0
                ):
                    torch.save(
                        model, f"{CHECKPOINT_FOLDER}/checkpoint{checkpoint_idx + 1}.pt"
                    )
                    save_count += 1

                if save_count % MODEL_SAVE_INTERVAL == 0:
                    checkpoint_idx += 1

            if epoch_loss_history[split] is None:
                epoch_loss_history[split] = {k: [] for k in batch_loss_history.keys()}

            for k, v in batch_loss_history.items():
                epoch_loss_history[split][k].append(sum(v) / len(v))

            match LOSS_LOG_MODE:
                case "console":
                    print(f"Epoch {epoch} / {split} : Loss {losses["total_loss"]}")
                case "file":
                    loss_log_file.write(
                        f"Epoch {epoch} / {split} : Loss {losses["total_loss"]}\n"
                    )

        print(f"Validation metrics for epoch {epoch}:")
        loss_log_file.write(f"Validation metrics for epoch {epoch}:\n")
        for metric, values in metrics:
            metrics[metric] = sum(values) / len(values)
            print(f"{metric}: {metrics[metric]}")
            loss_log_file.write(f"{metric}: {metrics[metric]}\n")
            metric_history[metric].append(metrics[metric])

        if LOSS_LOG_MODE == "graph":
            # chồng thêm validation loss lên plot của train loss
            for k, v_list in epoch_loss_history["validation"].items():
                epoch_ax.plot(v_list, label=k)
                epoch_ax.set_xlabel("Batch (Epoch progress)")
                epoch_ax.set_ylabel("Loss")
                epoch_ax.set_title("Loss Curves")
                epoch_ax.legend()
                epoch_ax.grid(True)
                plt.pause(0.01)
                epoch_ax.cla()

            # tạo 1 graph cho tất cả metrics
            for metric, v_list in metric_history.items():
                metric_ax.plot(v_list, label=k)
                metric_ax.set_xlabel("Batch (Epoch progress)")
                metric_ax.set_ylabel("Metric")
                metric_ax.set_title("Metric Curves")
                metric_ax.legend()
                metric_ax.grid(True)
                plt.pause(0.01)
                metric_ax.cla()
