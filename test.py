# from tokenizers.implementations import ByteLevelBPETokenizer
# from torch.utils.data import DataLoader

# from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
# from neural_intra_attention_model import NeuralIntraAttentionModel
# from pointer_generator_network import PointerGeneratorNetwork
# from tokenization import PointerGeneratorTokenizer
# from transformer import Transformer
# from utils import set_seed

# set_seed()

# # tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")
# tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")

# ds = CNNDailyMailDataset(
#     split="train",
#     tokenizer=tokenizer,
#     dataset_length=None,
# )
# loader = DataLoader(
#     ds,
#     collate_fn=collate_fn,
#     batch_sampler=DynamicBatchSampler(ds, max_tokens=10000, shuffle=True),
# )

# model = Transformer(
#     tokenizer=tokenizer,
#     d_model=128,
#     nhead=8,
#     num_layers=2,
#     learning_rate=1e-3,
#     device="cpu",
# )

# for batch_idx, batch in enumerate(loader):
#     batch_output_ids = model.infer(
#         batch["input_ids"],
#         max_output_length=10,
#         beam_width=4,
#         return_embedding=True,
#         return_attention=True,
#     )["output_ids"]
#     print(batch_output_ids)

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
                shuffle=True,
            ),
            pin_memory=True if DEVICE == "cuda" else False,
        ),
    }

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader["train"]):
            if batch_idx % 100 == 0:
                print(batch_idx)
            continue
