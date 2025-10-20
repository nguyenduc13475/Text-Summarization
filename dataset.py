import math
import os

import joblib
import requests
import torch
from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from utils import cache, text_to_token_ids


def load_dataset_manual():
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/datasets/abisee__cnn_dailymail/3.0.0"
    )
    os.makedirs(cache_dir, exist_ok=True)

    parquet_files = {
        "train": [
            "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/train-00000-of-00003.parquet",
            "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/train-00001-of-00003.parquet",
            "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/train-00002-of-00003.parquet",
        ],
        "validation": [
            "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/validation-00000-of-00001.parquet"
        ],
        "test": [
            "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/test-00000-of-00001.parquet"
        ],
    }

    def download_file(url, save_dir):
        local_path = os.path.join(save_dir, os.path.basename(url))
        if not os.path.exists(local_path):
            print(f"Downloading {url} ...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"{local_path} already exists, skipping download.")
        return local_path

    local_files = {}
    for split, urls in parquet_files.items():
        local_files[split] = [download_file(u, cache_dir) for u in urls]

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": local_files["train"],
            "validation": local_files["validation"],
            "test": local_files["test"],
        },
    )

    return dataset


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
        self.split = split
        if tokenizer is not None:
            self.vocab_size = self.tokenizer.get_vocab_size()

        if split in ["train", "cross validation", "test", "validation"]:
            if CNNDailyMailDataset.full_ds is None:
                try:
                    CNNDailyMailDataset.full_ds = load_dataset(
                        "abisee/cnn_dailymail", "3.0.0"
                    )
                except:
                    print(
                        "Failed to retrieve the dataset through the API, initiating direct HTTPS download as a fallback."
                    )
                    CNNDailyMailDataset.full_ds = load_dataset_manual()

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

        self.indices, self.lengths = cache(
            lambda: list(
                zip(
                    *sorted(
                        enumerate([len(sample["article"]) for sample in self.ds]),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
            ),
            f"{split}_sorted_indices.pkl",
        )

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
        target_ids = text_to_token_ids(self.tokenizer, hightlights, oov_list) + [
            self.tokenizer.token_to_id("</s>")
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "oov_list": oov_list,
            "input_text": article,
            "target_text": hightlights,
        }


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=10000):
        self.dataset = dataset
        self.max_tokens = max_tokens
        os.makedirs("cache", exist_ok=True)
        self.cache_batches_file = f"cache/{dataset.split}_batches.pkl"
        if os.path.exists(self.cache_batches_file):
            self.batches, self.start_idx = joblib.load(self.cache_batches_file)
            num_samples = sum([len(batch) for batch in self.batches])
            print(
                f"Load {dataset.split} batch indices successfully! ({num_samples} samples, {len(self.batches)} batches, continue at index {self.start_idx})"
            )
        else:
            self.batches = []
            self.start_idx = 0

    def update_batches(self, start_idx):
        joblib.dump((self.batches, start_idx), self.cache_batches_file)
        if os.path.exists("/content/drive/MyDrive"):
            joblib.dump(
                (self.batches, start_idx),
                f"/content/drive/MyDrive/{self.dataset.split}_batches.pkl",
            )

    def __iter__(self):
        max_input_length = 0
        max_target_length = 0

        for batch in self.batches:
            yield batch

        batch = []
        update_interval = -1
        for i, idx in enumerate(self.dataset.indices[self.start_idx :]):
            input_length = len(self.dataset[idx]["input_ids"])
            target_length = len(self.dataset[idx]["target_ids"])
            if target_length > input_length * 0.3:
                continue

            proj_max_input = max(max_input_length, input_length)
            proj_max_target = max(max_target_length, target_length)

            if batch and (
                proj_max_input * (len(batch) + 1) > self.max_tokens
                or proj_max_target * (len(batch) + 1) > self.max_tokens * 0.1
            ):
                self.batches.append(batch)
                if i > update_interval:
                    self.update_batches(self.start_idx + i)
                    update_interval += 100
                yield batch
                batch = []
                max_input_length = 0
                max_target_length = 0

            batch.append(idx)
            max_input_length = max(max_input_length, input_length)
            max_target_length = max(max_target_length, target_length)

        if batch:
            self.batches.append(batch)
            self.update_batches(len(self.dataset.indices))
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


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_sampler,
        collate_fn=None,
        skip_batches=0,
    ):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn or (lambda x: x)
        self.skip_batches = skip_batches

    def __iter__(self):
        self.batch_iter = iter(self.batch_sampler)

        for _ in range(self.skip_batches):
            try:
                next(self.batch_iter)
            except StopIteration:
                break

        return self

    def __next__(self):
        batch_indices = next(self.batch_iter)
        batch_samples = [self.dataset[idx] for idx in batch_indices]
        batch = self.collate_fn(batch_samples)

        return batch

    def __len__(self):
        return len(self.batch_sampler)
