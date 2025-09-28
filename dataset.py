import math
import random

import torch
from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


class CNNDailyMailDataset(Dataset):
    def __init__(
        self,
        split="train",
        vocab_file="vocab.json",
        merges_file="merges.txt",
        fold=None,
        num_folds=None,
        dataset=None,
    ):
        super().__init__()
        self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        self.vocab_size = self.tokenizer.get_vocab_size()

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

        if fold is not None and num_folds is not None:
            total_size = len(self.ds)
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
        article, summary = sample["article"], sample["highlights"]

        encoded_input = self.tokenizer.encode(article)
        encoded_target = self.tokenizer.encode(summary)

        input_ids = []
        labels = []
        oov_list = []

        for token_idx, token in zip(encoded_input.ids, encoded_input.tokens):
            if token_idx == self.tokenizer.token_to_id("<unk>"):
                token_oov_idx = oov_list.index(token)
                if token_oov_idx == -1:
                    input_ids.append(self.vocab_size + len(oov_list))
                    oov_list.append(token)
                else:
                    input_ids.append(self.vocab_size + token_oov_idx)
            else:
                input_ids.append(token_idx)

        for token_idx, token in zip(encoded_target.ids, encoded_target.tokens):
            if token_idx == self.tokenizer.token_to_id("<unk>"):
                token_oov_idx = oov_list.index(token)
                if token_oov_idx == -1:
                    labels.append(self.vocab_size + len(oov_list))
                    oov_list.append(token)
                else:
                    labels.append(self.vocab_size + token_oov_idx)
            else:
                labels.append(token_idx)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "oov_list": oov_list,
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

        batch, max_len = [], 0
        for idx in indices:
            length = len(self.dataset[idx]["input_ids"])
            new_max_len = max(max_len, length)

            if batch and new_max_len * (len(batch) + 1) > self.max_tokens:
                yield batch
                batch, max_len = [], 0

            batch.append(idx)
            max_len = max(max_len, length)
        if batch:
            yield batch

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    input_ids = [s["input_ids"] for s in batch]
    labels = [s["labels"] for s in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    input_lengths = (input_ids != 0).sum(dim=1)

    return {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "labels": labels,
        "oov_lists": [s["oov_list"] for s in batch],
    }
