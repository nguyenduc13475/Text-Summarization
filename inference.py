import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.manifold import TSNE
from tokenizers.implementations import ByteLevelBPETokenizer

from environment import adaptive_display, detect_runtime_env
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from text_rank import text_rank_summarize
from tokenization import PointerGeneratorTokenizer
from transformer import Transformer
from utils import load_checkpoint, text_to_token_ids, token_ids_to_text

MODELS = [
    # "TEXT_RANK",
    # "POINTER_GENERATOR_NETWORK",
    # "NEURAL_INTRA_ATTENTION_MODEL",
    "TRANSFORMER",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV = detect_runtime_env()
ATTENTION_PLOT = True
EMBEDDING_PLOT = True
# INPUT_TEXT = "Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization dataset."
INPUT_TEXT = load_dataset("abisee/cnn_dailymail", "3.0.0")["train"][151905]["article"]
print("Original text:")
print(INPUT_TEXT)
print("=" * 60)

if ENV in ("colab", "notebook"):
    from IPython.display import display


def find_latest_checkpoint(checkpoint_folder):
    if os.path.exists(checkpoint_folder) and any(
        re.match(r"^checkpoint_([0-9]\d*)\.pt$", f)
        for f in os.listdir(checkpoint_folder)
    ):
        latest_checkpoint = max(
            (
                int(m.group(1))
                for f in os.listdir(checkpoint_folder)
                if (m := re.match(r"^checkpoint_([0-9]\d*)\.pt$", f))
            )
        )

        return (
            f"{checkpoint_folder}/checkpoint_{latest_checkpoint}.pt",
            latest_checkpoint,
        )

    return None, None


def plot_attention_heatmap(
    ax,
    attention_distributions,
    output_tokens,
    input_tokens,
    xlabel,
    ylabel,
    title,
    title_font_size=16,
    axis_font_size=14,
    show_color_bar=True,
    display_immediately=True,
):
    im = ax.imshow(
        attention_distributions.detach().cpu().numpy(),
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )

    ax.set_xticks(range(len(input_tokens)))
    ax.set_xticklabels(
        input_tokens, rotation=90, fontsize=min(700 / len(input_tokens), 8)
    )
    ax.set_yticks(range(len(output_tokens)))
    ax.set_yticklabels(output_tokens, fontsize=min(200 / len(output_tokens), 8))
    ax.set_xlabel(xlabel, fontsize=axis_font_size)
    ax.set_ylabel(ylabel, fontsize=axis_font_size)
    ax.set_title(title, fontsize=title_font_size)

    if show_color_bar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention weight", fontsize=axis_font_size)
    if display_immediately:
        adaptive_display(figure, ENV)


def plot_tsne_embeddings(ax, embeddings, tokens, title, sample_size=300):
    X = embeddings.detach().cpu().numpy()
    X_unique, idx = np.unique(X, axis=0, return_index=True)
    tokens_unique = [tokens[i] for i in idx]

    if sample_size and len(tokens) > sample_size:
        idx = torch.randperm(len(tokens))[:sample_size].tolist()
        X = X[idx]
        tokens = [tokens[i] for i in idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    X_2d = tsne.fit_transform(X_unique)

    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=20, alpha=0.6)

    for i, token in enumerate(tokens_unique):
        ax.annotate(token, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)

    ax.set_title(title)
    adaptive_display(figure, ENV)


if __name__ == "__main__":
    for MODEL in MODELS:
        match MODEL:
            case "POINTER_GENERATOR_NETWORK":
                tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")
                model = PointerGeneratorNetwork(
                    tokenizer=tokenizer,
                    embedding_dim=128,
                    encoder_hidden_dim=256,
                    decoder_hidden_dim=256,
                    attention_dim=256,
                    bottle_neck_dim=512,
                    num_layers=2,
                    cov_loss_factor=1.0,
                    learning_rate=1e-2,
                    device=DEVICE,
                )
            case "NEURAL_INTRA_ATTENTION_MODEL":
                tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")
                model = NeuralIntraAttentionModel(
                    tokenizer=tokenizer,
                    embedding_dim=128,
                    hidden_dim=256,
                    bottle_neck_dim=512,
                    num_layers=2,
                    rl_loss_factor=0.0,
                    learning_rate=1e-3,
                    device=DEVICE,
                )
            case "TRANSFORMER":
                tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
                model = Transformer(
                    tokenizer=tokenizer,
                    d_model=256,
                    nhead=8,
                    num_layers=3,
                    learning_rate=1e-3,
                    device=DEVICE,
                )

        if MODEL == "TEXT_RANK":
            summary = text_rank_summarize(INPUT_TEXT, 3)[0]
        else:
            oov_list = []
            input_ids = text_to_token_ids(tokenizer, INPUT_TEXT, oov_list)
            checkpoint_file, checkpoint_idx = find_latest_checkpoint(
                f"{MODEL.lower()}_checkpoints"
            )
            if checkpoint_file is not None:
                load_checkpoint(model, checkpoint_file, map_location=DEVICE)
                print("Model loaded successfully!")

            output = model.infer(
                input_ids,
                max_output_length=200,
                beam_width=1,
                return_attention=ATTENTION_PLOT,
                return_embedding=EMBEDDING_PLOT,
            )

            summary, output_tokens = token_ids_to_text(
                tokenizer,
                output["output_ids"][0],
                oov_list,
                return_output="both",
            )

            input_tokens = token_ids_to_text(
                tokenizer,
                input_ids,
                oov_list,
                return_output="list",
            )

        print(f"Summary using {MODEL}:")
        print("=================================================")
        print(summary)
        print("=================================================")

        if ATTENTION_PLOT:
            match MODEL:
                case "POINTER_GENERATOR_NETWORK":
                    figure, ax = plt.subplots(figsize=(10, 10))

                    plot_attention_heatmap(
                        ax,
                        output["cross_attention_distributions"][0][
                            : len(output_tokens)
                        ],
                        output_tokens,
                        input_tokens,
                        "Input tokens",
                        "Generated summary tokens",
                        "Pointer Generator Cross Attention Heatmap (Summary -> Input)",
                    )
                case "NEURAL_INTRA_ATTENTION_MODEL":
                    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

                    plot_attention_heatmap(
                        axes[0],
                        output["cross_attention_distributions"][0][
                            : len(output_tokens)
                        ],
                        output_tokens,
                        input_tokens,
                        "Input tokens",
                        "Generated summary tokens",
                        "Neural Intra Attention Heatmap (Summary -> Input)",
                        title_font_size=10,
                        display_immediately=(ENV == "gui"),
                    )

                    plot_attention_heatmap(
                        axes[1],
                        output["decoder_attention_distributions"][0][
                            : len(output_tokens) - 1, : len(output_tokens) - 1
                        ],
                        output_tokens[1:],
                        output_tokens[:-1],
                        "Generated summary tokens",
                        "Generated summary tokens",
                        "Neural Intra Attention Heatmap (Summary -> Summary)",
                        title_font_size=10,
                        display_immediately=(ENV == "gui"),
                    )
                    figure.tight_layout(pad=2)
                    adaptive_display(figure, ENV)

                case "TRANSFORMER":
                    num_heads = len(output["encoder_self_attention_distributions"][0])
                    cols = math.ceil(math.sqrt(num_heads))
                    rows = math.ceil(num_heads / cols)
                    figure = plt.figure(figsize=(cols * 5, rows * 5))
                    for i, attention_head in enumerate(
                        output["encoder_self_attention_distributions"][0]
                    ):
                        ax = figure.add_subplot(rows, cols, i + 1)
                        plot_attention_heatmap(
                            ax,
                            attention_head,
                            input_tokens,
                            input_tokens,
                            "Input tokens",
                            "Input tokens",
                            f"Transformer Encoder Self-Attention - Head {i}",
                            title_font_size=7,
                            axis_font_size=7,
                            show_color_bar=i % cols == cols - 1 or i == num_heads - 1,
                            display_immediately=(ENV == "gui"),
                        )
                    figure.tight_layout(pad=4)
                    adaptive_display(figure, ENV)

                    figure = plt.figure(figsize=(cols * 5, rows * 5))
                    for i, attention_head in enumerate(
                        output["decoder_self_attention_distributions"][0]
                    ):
                        ax = figure.add_subplot(rows, cols, i + 1)
                        plot_attention_heatmap(
                            ax,
                            attention_head,
                            output_tokens,
                            output_tokens,
                            "Generated summary tokens",
                            "Generated summary tokens",
                            f"Transformer Decoder Self-Attention - Head {i}",
                            title_font_size=7,
                            axis_font_size=7,
                            show_color_bar=i % cols == cols - 1 or i == num_heads - 1,
                            display_immediately=(ENV == "gui"),
                        )
                    figure.tight_layout(pad=4)
                    adaptive_display(figure, ENV)

                    figure = plt.figure(figsize=(cols * 5, rows * 5))
                    for i, attention_head in enumerate(
                        output["cross_attention_distributions"][0]
                    ):
                        ax = figure.add_subplot(rows, cols, i + 1)
                        plot_attention_heatmap(
                            ax,
                            attention_head,
                            output_tokens,
                            input_tokens,
                            "Input tokens",
                            "Generated summary tokens",
                            f"Transformer Cross Attention - Head {i}",
                            title_font_size=7,
                            axis_font_size=7,
                            show_color_bar=i % cols == cols - 1 or i == num_heads - 1,
                            display_immediately=(ENV == "gui"),
                        )
                    figure.tight_layout(pad=4)
                    adaptive_display(figure, ENV)

        if EMBEDDING_PLOT and MODEL != "TEXT_RANK":
            figure, ax = plt.subplots(figsize=(10, 10))
            plot_tsne_embeddings(
                ax,
                output["input_embeddings"][0],
                input_tokens,
                f"T-SNE projection of {MODEL} input embeddings",
            )

    if ENV == "gui":
        plt.show()
