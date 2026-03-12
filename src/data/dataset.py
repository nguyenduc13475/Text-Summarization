import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, TypedDict

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.data import Dataset, Sampler

from src.utils.logger import setup_logger
from src.utils.utils import cache, text_to_token_ids

# Setup local logger for data pipeline tracking
logger = setup_logger(name="DataPipeline", log_file="dataset.log")


class BatchData(TypedDict):
    """
    Standardized batch structure for the summarization pipeline.
    Ensures type safety across Trainer, Model, and Evaluator.

    Attributes:
        input_ids: Encoded source text tokens.
        input_length: Actual lengths of source sequences (before padding).
        input_text: Raw source strings (for visualization/TextRank).
        target_ids: Encoded reference summary tokens.
        target_length: Actual lengths of target sequences.
        target_text: Raw reference strings.
        oov_list: Tokens not in vocab, used by Pointer-Generator mechanism.
    """

    input_ids: torch.Tensor
    input_length: torch.Tensor
    input_text: List[str]
    target_ids: torch.Tensor
    target_length: torch.Tensor
    target_text: List[str]
    oov_list: List[List[str]]


class CNNDailyMailDataset(Dataset):
    """
    Production-grade dataset wrapper for CNN/DailyMail.
    Implements singleton pattern for dataset loading and efficient caching.
    """

    full_ds: Optional[HFDataset] = None

    def __init__(
        self,
        split: str = "train",
        tokenizer: Optional[Any] = None,
        fold: Optional[int] = None,
        num_folds: Optional[int] = None,
    ):
        """
        Initializes the dataset with support for Cross-Validation slicing.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            tokenizer: Tokenizer instance for text encoding.
            fold: Current fold index for CV.
            num_folds: Total number of folds for CV.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split

        # Initialize dataset once to optimize memory footprint
        if CNNDailyMailDataset.full_ds is None:
            self._initialize_dataset()

        # Map logic for cross-validation splits
        split_map = {
            "train": "train",
            "cross validation": "train",
            "test": "test",
            "validation": "validation",
        }
        target_split = split_map.get(split, "train")
        raw_ds = CNNDailyMailDataset.full_ds[target_split]

        # Handle Cross-Validation partitioning
        if fold is not None and num_folds is not None:
            total_size = len(raw_ds)
            fold_size = math.ceil(total_size / num_folds)
            indices = list(range(total_size))
            start_idx = fold * fold_size
            end_idx = min(start_idx + fold_size, total_size)

            if split == "train":
                self.ds = raw_ds.select(indices[:start_idx] + indices[end_idx:])
            else:
                self.ds = raw_ds.select(indices[start_idx:end_idx])
        else:
            self.ds = raw_ds

        # Pre-calculate sequence lengths for Dynamic Batching efficiency
        self.indices, _ = cache(
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

    def _initialize_dataset(self) -> None:
        """Loads the dataset from HuggingFace with fallback logging."""
        try:
            CNNDailyMailDataset.full_ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
            logger.info("Successfully loaded CNN/DailyMail dataset.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise e

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch and tokenize a single data point."""
        sample = self.ds[idx]
        article, highlights = sample["article"], sample["highlights"]

        if self.tokenizer is None:
            return {"input_text": article, "target_text": highlights}

        oov_list = []
        input_ids = text_to_token_ids(self.tokenizer, article, oov_list)
        target_ids = text_to_token_ids(self.tokenizer, highlights, oov_list) + [
            self.tokenizer.token_to_id("</s>")
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "oov_list": oov_list,
            "input_text": article,
            "target_text": highlights,
        }


def build_collate_fn(tokenizer: Any) -> Callable[[List[Dict[str, Any]]], BatchData]:
    """
    Factory to create a collate function that handles dynamic padding.
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> BatchData:
        pad_id = tokenizer.token_to_id("<pad>")

        # Dynamic padding to the longest sequence in the current batch only
        input_ids_padded = pad_sequence(
            [s["input_ids"] for s in batch], batch_first=True, padding_value=pad_id
        )
        target_ids_padded = pad_sequence(
            [s["target_ids"] for s in batch], batch_first=True, padding_value=pad_id
        )

        return {
            "input_ids": input_ids_padded,
            "input_length": (input_ids_padded != pad_id).sum(dim=1),
            "input_text": [s["input_text"] for s in batch],
            "target_ids": target_ids_padded,
            "target_length": (target_ids_padded != pad_id).sum(dim=1),
            "target_text": [s["target_text"] for s in batch],
            "oov_list": [s["oov_list"] for s in batch],
        }

    return collate_fn


class DynamicBatchSampler(Sampler[List[int]]):
    """
    Groups sequences of similar lengths into batches to minimize padding.
    Crucial for training large Transformer models efficiently.
    """

    def __init__(
        self,
        dataset: CNNDailyMailDataset,
        max_tokens: int = 10000,
        batch_nums: Optional[int] = None,
        start_batch: int = 0,
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.batches: List[List[int]] = []
        self._build_batches()

        if batch_nums is not None:
            self.batches = self.batches[start_batch : start_batch + batch_nums]

    def _build_batches(self) -> None:
        curr_batch: List[int] = []
        max_len_in_batch = 0

        for idx in self.dataset.indices:
            # Estimate token length to fit into GPU VRAM constraints
            # (Using a safe estimation or actual length if available)
            token_len = len(self.dataset[idx]["input_ids"])
            new_max_len = max(max_len_in_batch, token_len)

            if new_max_len * (len(curr_batch) + 1) <= self.max_tokens:
                curr_batch.append(idx)
                max_len_in_batch = new_max_len
            else:
                if curr_batch:
                    self.batches.append(curr_batch)
                curr_batch = [idx]
                max_len_in_batch = token_len

        if curr_batch:
            self.batches.append(curr_batch)

    def __iter__(self) -> Iterator[BatchData]:
        import random

        # Shuffle batch order for training to ensure stochasticity
        if self.dataset.split == "train":
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


class SummaryDataLoader(PyTorchDataLoader):
    """
    Custom wrapper for PyTorch DataLoader to handle skip-logic
    needed for interrupted training sessions.
    """

    def __init__(self, *args: Any, skip_batches: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_batches = skip_batches

    def __iter__(self) -> Iterator[BatchData]:
        it = super().__iter__()
        for _ in range(self.skip_batches):
            next(it)
        return it
