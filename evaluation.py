import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from metrics import compute_metric
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from text_rank import text_rank_summarize
from transformer import Transformer
from utils import token_ids_to_text

MODELS = [
    "TEXT_RANK",
    "POINTER_GENERATOR_NETWORK",
    "NEURAL_INTRA_ATTENTION_MODEL",
    "TRANSFORMER",
]  # có thể rút về mảng 1 phần tử nếu không muốn so sánh
DEVICE = "cpu"
METRICS = ["rouge1", "rouge2", "rougeL", "bleu", "meteor", "bertscore", "moverscore"]
MAX_TOKENS_EACH_BATCH = 10000


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
    test_ds = CNNDailyMailDataset(split="test")
    test_loader = DataLoader(
        test_ds,
        collate_fn=collate_fn,
        batch_sampler=DynamicBatchSampler(
            test_ds, max_tokens=MAX_TOKENS_EACH_BATCH, shuffle=True
        ),
    )

    metrics = dict()
    for MODEL in MODELS:
        if MODEL != "TEXT_RANK":
            checkpoint_file, checkpoint_idx = find_latest_checkpoint(
                f"{model.lower()}_checkpoints"
            )
            model = torch.load(checkpoint_file)

        metrics[MODEL] = dict()
        for metric in METRICS:
            metrics[MODEL][metric] = []
            for batch_idx, batch in enumerate(test_loader):
                for sample_idx in len(batch["input_ids"]):
                    if MODEL == "TEXT_RANK":
                        article = token_ids_to_text(
                            test_ds.tokenizer,
                            batch["input_ids"][sample_idx],
                            batch["oov_lists"][sample_idx],
                            test_ds.vocab_size,
                        )
                        candidate_summary = text_rank_summarize(article, 3)
                    else:
                        output_ids = model.infer(batch["input_ids"][sample_idx])
                        candidate_summary = token_ids_to_text(
                            test_ds.tokenizer,
                            output_ids,
                            batch["oov_lists"][sample_idx],
                            test_ds.vocab_size,
                        )
                    reference_summary = token_ids_to_text(
                        test_ds.tokenizer,
                        batch["labels"][sample_idx],
                        batch["oov_lists"][sample_idx],
                        test_ds.vocab_size,
                    )

                    for metric in METRICS:
                        metrics[MODEL][metric].append(
                            compute_metric(metric, candidate_summary, reference_summary)
                        )

        print(f"Validation metrics for {MODEL}:")
        for metric, values in metrics[MODEL]:
            metrics[MODEL][metric] = sum(values) / len(values)
            print(f"{metric}: {metrics[MODEL][metric]}")

    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, MODEL in enumerate(MODELS):
        values = [metrics[MODEL][metric] for metric in METRICS]
        ax.bar(x + i * width, values, width, label=MODEL)

    ax.set_xticks(x + width)
    ax.set_xticklabels(METRICS, rotation=30)
    ax.set_ylabel("Score")
    ax.set_title("Metrics comparison on test set")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
