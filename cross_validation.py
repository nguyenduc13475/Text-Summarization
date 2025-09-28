import math
import os
import re
import shutil

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from transformer import Transformer

MODEL = "TRANSFORMER"
NUM_EPOCHS = 1000
MAX_TOKENS_EACH_BATCH = 10000
DEVICE = "cpu"
NUM_FOLDS = 10
LOSS_LOG_MODE = "console"
LOSS_LOG_INTERVAL = 10

if __name__ == "__main__":
    loss_log_file = open("loss_log.txt", "w")
    loss_histories = []
    cross_validation_losses = []
    for fold in range(NUM_FOLDS):
        if LOSS_LOG_MODE == "graph":
            plt.clf()
            plt.ion()
            n = fold + 1
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axes = axes.flatten()
            for i, loss_history in enumerate(loss_histories):
                axes[i].plot(loss_history)
                axes[i].set_title(f"Fold {i}")

        for split in ["train", "cross validate"]:
            ds = CNNDailyMailDataset(split=split, fold=fold, num_folds=NUM_FOLDS)
            loader = DataLoader(
                ds,
                collate_fn=collate_fn,
                batch_sampler=DynamicBatchSampler(
                    ds, max_tokens=MAX_TOKENS_EACH_BATCH, shuffle=True
                ),
            )

            match MODEL:
                case "POINTER_GENERATOR_NETWORK":
                    model = PointerGeneratorNetwork(
                        vocab_size=ds.vocab_size,
                        tokenizer=ds.tokenizer,
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
                        vocab_size=ds.vocab_size,
                        tokenizer=ds.tokenizer,
                        embedding_dim=128,
                        hidden_dim=160,
                        unknown_token=ds.tokenizer.token_to_id("<unk>"),
                        start_token=ds.tokenizer.token_to_id("<s>"),
                        end_token=ds.tokenizer.token_to_id("</s>"),
                        device=DEVICE,
                    )
                case "TRANSFORMER":
                    model = Transformer(
                        vocab_size=ds.vocab_size,
                        tokenizer=ds.tokenizer,
                        d_model=128,
                        nhead=8,
                        num_layers=2,
                        learning_rate=1e-3,
                        device=DEVICE,
                    )

            loss_history = None
            caller = (
                model.train_one_batch if split == "train" else model.validate_one_batch
            )
            for epoch in range(NUM_EPOCHS):
                for batch_idx, batch in enumerate(loader):

                    match MODEL:
                        case "POINTER_GENERATOR_NETWORK":
                            losses = caller(
                                batch["input_ids"],
                                batch["labels"],
                                batch["input_lengths"],
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
                                batch["input_ids"],
                                batch["labels"],
                                batch["input_lengths"],
                            )

                    if loss_history is None:
                        loss_history = {k: [] for k in losses.keys()}

                    for k, v in losses.items():
                        loss_history[k].append(v.item())

                    if (
                        split == "train"
                        and LOSS_LOG_INTERVAL is not None
                        and batch_idx % LOSS_LOG_INTERVAL == 0
                    ):
                        match LOSS_LOG_MODE:
                            case "console":
                                print(
                                    f"Fold {fold} / Epoch {epoch} / Batch {batch_idx} : Loss {losses["total_loss"]}"
                                )
                            case "file":
                                loss_log_file.write(
                                    f"Fold {fold} / Epoch {epoch} / Batch {batch_idx} : Loss {losses["total_loss"]}\n"
                                )
                            case "graph":
                                for k, v_list in loss_history.items():
                                    axes[-1].plot(v_list, label=k)
                                axes[-1].set_xlabel("Batch (Epoch progress)")
                                axes[-1].set_ylabel("Loss")
                                axes[-1].set_title(f"Fold {fold} Loss Curves")
                                axes[-1].legend()
                                axes[-1].grid(True)
                                plt.pause(0.01)
                                axes[-1].cla()

                average_loss = sum(loss_history["total_loss"]) / (batch_idx + 1)
                match LOSS_LOG_MODE:
                    case "console":
                        print(
                            f"Fold {fold} / Epoch {epoch} : Average Loss {average_loss}"
                        )
                    case "file":
                        loss_log_file.write(
                            f"Fold {fold} / Epoch {epoch} : Average Loss {average_loss}\n"
                        )

            if split == "validate":
                cross_validation_loss = average_loss
                cross_validation_losses.append(cross_validation_loss)
            loss_histories.append(loss_history)

        match LOSS_LOG_MODE:
            case "console":
                print(f"Fold {fold} : Validate Loss {cross_validation_loss}")

        loss_log_file.write(f"Fold {fold} : Validate Loss {cross_validation_loss}\n")

    print(
        f"Average loss on all fold is: {sum(cross_validation_losses) / len(cross_validation_losses)}"
    )
    loss_log_file.close()
    if LOSS_LOG_MODE == "graph":
        plt.ioff()
        plt.show()
