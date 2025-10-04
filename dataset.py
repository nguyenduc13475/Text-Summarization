import math
import random

import torch
from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from utils import text_to_token_ids


class CNNDailyMailDataset(Dataset):
    full_ds = None

    def __init__(
        self,
        split="train",
        tokenizer=ByteLevelBPETokenizer("vocab.json", "merges.txt"),
        fold=None,
        num_folds=None,
        dataset=None,
        dataset_length=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.vocab_size = self.tokenizer.get_vocab_size()

        if split in ["train", "cross validation", "test", "validation"]:
            if CNNDailyMailDataset.full_ds is None:
                CNNDailyMailDataset.full_ds = load_dataset(
                    "abisee/cnn_dailymail", "3.0.0"
                )

        full_ds = CNNDailyMailDataset.full_ds

        match split:
            case "train" | "cross validation":
                full_ds = full_ds["train"]
            case "test":
                full_ds = full_ds["test"]
            case "validation":
                full_ds = full_ds["validation"]
            case _:
                if dataset is not None:
                    full_ds = dataset
                else:
                    raise ValueError("at least split or dataset must be valid!")

        if dataset_length is not None:
            full_ds = full_ds.select(range(dataset_length))

        if fold is not None and num_folds is not None:
            total_size = len(full_ds)
            fold_size = math.ceil(total_size / num_folds)
            indices = list(range(total_size))

            start_idx = fold * fold_size
            end_idx = min(start_idx + fold_size, total_size)

            if split == "train":
                self.ds = full_ds.select(indices[:start_idx] + indices[end_idx:])
            else:
                self.ds = full_ds.select(indices[start_idx:end_idx])
        else:
            self.ds = full_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        article, hightlights = sample["article"], sample["highlights"]

        if self.tokenizer is None:
            return {
                "input_text": article,
                "target_text": hightlights,
            }

        oov_list = []
        input_ids = text_to_token_ids(self.tokenizer, article, oov_list)
        target_ids = text_to_token_ids(
            self.tokenizer, hightlights, oov_list
        ) + self.tokenizer.token_to_id("</s>")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "oov_list": oov_list,
            "input_text": article,
            "target_text": hightlights,
        }


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=10000, shuffle=True):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch, max_input_length = [], 0
        for idx in indices:
            input_length = len(self.dataset[idx]["input_ids"])
            new_max_input_length = max(max_input_length, input_length)

            if batch and new_max_input_length * (len(batch) + 1) > self.max_tokens:
                yield batch
                batch, max_input_length = [], 0

            batch.append(idx)
            max_input_length = max(max_input_length, input_length)
        if batch:
            yield batch

    def __len__(self):
        return len(self.dataset)


def build_collate_fn(tokenizer):
    def collate_fn(batch):
        pad_token = tokenizer.token_to_id("<pad>")
        batch_input_ids = [sample["input_ids"] for sample in batch]
        batch_target_ids = [sample["target_ids"] for sample in batch]

        batch_input_ids = pad_sequence(
            batch_input_ids, batch_first=True, padding_value=pad_token
        )
        batch_target_ids = pad_sequence(
            batch_target_ids, batch_first=True, padding_value=pad_token
        )

        input_lengths = (batch_input_ids != pad_token).sum(dim=1)
        target_lengths = (batch_target_ids != pad_token).sum(dim=1)

        return {
            "input_ids": batch_input_ids,
            "input_length": input_lengths,
            "input_text": [sample["input_text"] for sample in batch],
            "target_ids": batch_target_ids,
            "target_length": target_lengths,
            "target_text": [sample["target_text"] for sample in batch],
            "oov_list": [sample["oov_list"] for sample in batch],
        }

    return collate_fn
