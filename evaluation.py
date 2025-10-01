import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from environment import detect_runtime_env, try_set_window_position
from metrics import compute_metric
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from text_rank import text_rank_summarize
from tokenization import PointerGeneratorTokenizer
from transformer import Transformer
from utils import load_checkpoint, set_seed, token_ids_to_text

set_seed()

MODELS = [
    "TEXT_RANK",
    "POINTER_GENERATOR_NETWORK",
    "NEURAL_INTRA_ATTENTION_MODEL",
    "TRANSFORMER",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRICS = ["rouge1", "rouge2", "rougeL", "bleu4", "meteor"]
MAX_TOKENS_EACH_BATCH = 10000
DATASET_LENGTH = 15
ENV = detect_runtime_env()
if ENV in ("colab", "notebook"):
    from IPython.display import display


def find_latest_checkpoint(checkpoint_folder):
    if os.path.exists(checkpoint_folder) and any(
        re.match(r"^checkpoint([1-9]\d*)\.pt$", f)
        for f in os.listdir(checkpoint_folder)
    ):
        latest_checkpoint = max(
            (
                int(m.group(1))
                for f in os.listdir(checkpoint_folder)
                if (m := re.match(r"^checkpoint([1-9]\d*)\.pt$", f))
            )
        )

        return (
            f"{checkpoint_folder}/checkpoint{latest_checkpoint}.pt",
            latest_checkpoint,
        )

    return None, None


if __name__ == "__main__":
    metrics = defaultdict(lambda: defaultdict(list))
    for MODEL in MODELS:
        match MODEL:
            case "POINTER_GENERATOR_NETWORK":
                tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")
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
                tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")
                model = NeuralIntraAttentionModel(
                    tokenizer=tokenizer,
                    embedding_dim=128,
                    hidden_dim=160,
                    rl_loss_factor=0.75,
                    learning_rate=1e-4,
                    device=DEVICE,
                )
            case "TRANSFORMER":
                tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
                model = Transformer(
                    tokenizer=tokenizer,
                    d_model=128,
                    nhead=8,
                    num_layers=2,
                    learning_rate=1e-3,
                    device=DEVICE,
                )

        if MODEL == "TEXT_RANK":
            test_ds = CNNDailyMailDataset(
                split="test", tokenizer=None, dataset_length=DATASET_LENGTH
            )
            test_loader = DataLoader(test_ds, batch_size=2)
        else:
            test_ds = CNNDailyMailDataset(
                split="test", tokenizer=tokenizer, dataset_length=DATASET_LENGTH
            )
            test_loader = DataLoader(
                test_ds,
                collate_fn=collate_fn,
                batch_sampler=(
                    DynamicBatchSampler(
                        test_ds, max_tokens=MAX_TOKENS_EACH_BATCH, shuffle=True
                    )
                ),
            )
            checkpoint_file, checkpoint_idx = find_latest_checkpoint(
                f"{MODEL.lower()}_checkpoints"
            )
            if checkpoint_file is not None:
                load_checkpoint(model, checkpoint_file, map_location=DEVICE)
                print("Model loaded successfully!")

        for batch_idx, batch in enumerate(test_loader):
            if MODEL == "TEXT_RANK":
                candidate_summaries = text_rank_summarize(batch["input_text"], 3)
                reference_summaries = batch["target_text"]
            else:
                batch_output_ids = model.infer(
                    batch["input_ids"], max_output_length=5, beam_width=2
                )["output_ids"]
                candidate_summaries = [
                    token_ids_to_text(
                        tokenizer,
                        output_ids,
                        oov_list,
                    )
                    for output_ids, oov_list in zip(batch_output_ids, batch["oov_list"])
                ]
                reference_summaries = [
                    token_ids_to_text(
                        tokenizer,
                        target_ids,
                        oov_list,
                    )
                    for target_ids, oov_list in zip(
                        batch["target_ids"], batch["oov_list"]
                    )
                ]

            for metric in METRICS:
                for candidate_summary, reference_summary in zip(
                    candidate_summaries, reference_summaries
                ):
                    metrics[MODEL][metric].append(
                        compute_metric(metric, candidate_summary, reference_summary)
                    )

        print("=================================================")
        print(f"Validation metrics for {MODEL}:")
        for metric, values in metrics[MODEL].items():
            metrics[MODEL][metric] = sum(values) / len(values)
            print(f"{metric.upper()}: {metrics[MODEL][metric]}")
        print("=================================================")

    x = np.arange(len(METRICS)) * 1.5
    width = 0.25
    figure, ax = plt.subplots(figsize=(12, 6))
    for i, MODEL in enumerate(MODELS):
        values = [metrics[MODEL][metric] for metric in METRICS]
        ax.bar(x + i * width, values, width, label=MODEL)

    ax.set_xticks(x + 0.75)
    ax.set_xticklabels([metric.upper() for metric in METRICS])
    ax.set_ylabel("Score")
    ax.set_title("Metrics comparison on test set")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    try_set_window_position(0, 0)
    match ENV:
        case "colab" | "notebook":
            display(figure)
        case "gui":
            plt.show()
