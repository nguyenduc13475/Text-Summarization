import os
import re

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, collate_fn
from text_rank import text_rank_summarize
from utils import pad_and_stack, token_ids_to_text

MODELS = [
    "TEXT_RANK",
    "POINTER_GENERATOR_NETWORK",
    "NEURAL_INTRA_ATTENTION_MODEL",
    "TRANSFORMER",
]
DEVICE = "cpu"
ATTENTION_PLOT = True
EMBEDDING_PLOT = True

INPUT_TEXT = "I have a chicken."


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


def plot_attention_heatmap(attention_distributions, output_tokens, input_tokens):
    attn_matrix = attention_distributions.detach().cpu().numpy()

    plt.figure(figsize=(max(6, len(input_tokens) * 0.5), max(4, len(summary) * 0.5)))
    plt.imshow(attn_matrix, aspect="auto", cmap="viridis")

    plt.xticks(range(len(input_tokens)), input_tokens, rotation=90)
    plt.yticks(range(len(output_tokens)), output_tokens)

    plt.xlabel("Input tokens")
    plt.ylabel("Generated summary tokens")
    plt.colorbar(label="Attention weight")
    plt.title("Attention Heatmap (Summary → Input)")
    plt.tight_layout()
    plt.show()


def plot_tsne_embeddings(embeddings, tokens, sample_size=300):
    X = embeddings.detach().cpu().numpy()

    if sample_size and len(tokens) > sample_size:
        idx = torch.randperm(len(tokens))[:sample_size].tolist()
        X = X[idx]
        tokens = [tokens[i] for i in idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=20, alpha=0.6)

    if len(tokens) <= 50:
        for i, token in enumerate(tokens):
            plt.annotate(token, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)

    plt.title("t-SNE of Token Embeddings")
    plt.show()


if __name__ == "__main__":
    singleton_ds = CNNDailyMailDataset(
        split=None, dataset=[{"article": INPUT_TEXT, "highlights": ""}]
    )
    singleton_loader = DataLoader(
        singleton_ds,
        collate_fn=collate_fn,
    )
    singleton_batch = next(iter(singleton_loader))
    input_ids = singleton_batch["input_ids"][0]
    oov_list = singleton_batch["oov_lists"][0]

    for MODEL in MODELS:
        if MODEL != "TEXT_RANK":
            checkpoint_file, checkpoint_idx = find_latest_checkpoint(
                f"{model.lower()}_checkpoints"
            )
            model = torch.load(checkpoint_file)

        if MODEL == "TEXT_RANK":
            summary = text_rank_summarize(INPUT_TEXT, 3)
        else:
            output = model.infer(
                input_ids,
                return_attention=ATTENTION_PLOT,
                return_embedding=EMBEDDING_PLOT,
            )
            output_ids = output[0]
            summary, output_tokens = token_ids_to_text(
                singleton_ds.tokenizer,
                output_ids,
                oov_list,
                return_output="both",
            )
            input_tokens = token_ids_to_text(
                singleton_ds.tokenizer,
                input_ids,
                oov_list,
                return_output="list",
            )

        print(f"Summary for model {MODEL}:\n{summary}")
        if ATTENTION_PLOT:
            match MODEL:
                case "POINTER_GENERATOR_NETWORK":
                    plot_attention_heatmap(
                        pad_and_stack(output["input_attention_distributions"]),
                        output_tokens,
                        input_tokens,
                    )

                case "NEURAL_INTRA_ATTENTION_MODEL":
                    plot_attention_heatmap(
                        pad_and_stack(output["input_attention_distributions"]),
                        output_tokens,
                        input_tokens,
                    )

                    plot_attention_heatmap(
                        pad_and_stack(output["output_attention_distributions"]),
                        output_tokens,
                        output_tokens,
                    )
                case "TRANSFORMERS":
                    plot_attention_heatmap(
                        pad_and_stack(output["input_attention_distributions"]),
                        input_tokens,
                        input_tokens,
                    )

                    plot_attention_heatmap(
                        pad_and_stack(output["output_attention_distributions"]),
                        output_tokens,
                        output_tokens,
                    )

                    plot_attention_heatmap(
                        pad_and_stack(output["cross_attention_distributions"]),
                        output_tokens,
                        input_tokens,
                    )

        if EMBEDDING_PLOT and MODEL != "TEXT_RANK":
            plot_tsne_embeddings(output["embedding"], input_tokens)
