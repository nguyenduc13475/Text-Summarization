from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tokenizers.implementations import ByteLevelBPETokenizer

from dataset import (
    CNNDailyMailDataset,
    DataLoader,
    DynamicBatchSampler,
    build_collate_fn,
)
from environment import adaptive_display, detect_runtime_env, try_set_window_position
from metrics import compute_metric
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from tokenization import PointerGeneratorTokenizer
from transformer import Transformer
from utils import load_checkpoint, set_seed, token_ids_to_text

set_seed()

MODEL = "POINTER_GENERATOR_NETWORK"
CHECKPOINT_FOLDER = f"{MODEL.lower()}_checkpoints"
NUM_EPOCHS = 200
MAX_TOKENS_EACH_BATCH = 32000
VALIDATION_DATASET_LENGTH = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV = detect_runtime_env()
METRICS = ["rouge1", "rouge2", "rougeL", "bleu4", "meteor", "bertscore", "moverscore"]

if ENV in ("colab", "notebook"):
    from IPython.display import clear_output, display

if __name__ == "__main__":
    figure, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Metric Per Sample")
    ax.set_title(f"Metric Curves")
    ax.grid(True)
    try_set_window_position(0, 0)
    figure.tight_layout(pad=2.0)
    adaptive_display(figure, ENV)
    metric_line_2ds = defaultdict(lambda: None)

    if MODEL == "TRANSFORMER":
        tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    else:
        tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")

    collate_fn = build_collate_fn(tokenizer)
    ds = CNNDailyMailDataset(
        split="validation",
        tokenizer=tokenizer,
        dataset_length=VALIDATION_DATASET_LENGTH,
    )
    loader = (
        DataLoader(
            ds,
            collate_fn=collate_fn,
            batch_sampler=DynamicBatchSampler(
                ds,
                max_tokens=MAX_TOKENS_EACH_BATCH,
            ),
            pin_memory=True if DEVICE == "cuda" else False,
        ),
    )

    match MODEL:
        case "POINTER_GENERATOR_NETWORK":
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
            model = NeuralIntraAttentionModel(
                tokenizer=tokenizer,
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2,
                rl_loss_factor=0.75,
                learning_rate=1e-2,
                device=DEVICE,
            )
        case "TRANSFORMER":
            model = Transformer(
                tokenizer=tokenizer,
                d_model=256,
                nhead=8,
                num_layers=3,
                learning_rate=1e-3,
                device=DEVICE,
            )

    metric_history = defaultdict(list)
    for epoch in range(NUM_EPOCHS):
        try:
            load_checkpoint(
                model, f"{CHECKPOINT_FOLDER}/checkpoint_{epoch}.pt", map_location=DEVICE
            )
        except:
            continue
        print(f"Checkpoint {epoch} loaded successfully!")
        metrics = defaultdict(list)
        num_samples = 0
        for batch_idx, batch in enumerate(loader):
            num_samples += len(batch["input_ids"])
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
                for output_ids, oov_list in zip(batch_output_ids, batch["oov_list"])
            ]
            for metric, values in compute_metric(
                METRICS, output_texts, batch["target_text"]
            ).items():
                metrics[metric].extend(values)
            print(
                f"Validated {num_samples}/{VALIDATION_DATASET_LENGTH if VALIDATION_DATASET_LENGTH is not None else len(ds)} samples"
            )
            torch.cuda.empty_cache()

        print("=================================================")
        print(f"Validation metrics at epoch {epoch} :")
        for metric, values in metrics.items():
            metrics[metric] = sum(values) / len(values)
            print(f"{metric.upper()} : {metrics[metric]}")
            metric_history[metric].append(metrics[metric])
        print("=================================================")

        for metric, values in metric_history.items():
            for (
                metric,
                values,
            ) in metric_history.items():
                if metric_line_2ds[metric] is None:
                    metric_line_2ds[metric] = ax.plot(values, label=metric.upper())[0]
                    ax.legend()
                else:
                    metric_line_2ds[metric].set_xdata(range(len(values)))
                    metric_line_2ds[metric].set_ydata(values)

            ax.relim()
            ax.autoscale()
            figure.canvas.draw()
            figure.canvas.flush_events()
            if ENV in ("colab", "notebook"):
                clear_output(wait=True)
                display(figure)

    if ENV == "gui":
        plt.show()
