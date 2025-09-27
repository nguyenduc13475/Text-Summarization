from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from neural_intra_attention import NeuralIntraAttention
from pointer_generator import PointerGenerator
from transformer import TransformerSummarizer

MODEL = "TRANSFORMER"

if __name__ == "__main__":
    train_ds = CNNDailyMailDataset()
    train_loader = DataLoader(
        train_ds,
        collate_fn=collate_fn,
        batch_sampler=DynamicBatchSampler(train_ds, max_tokens=10000, shuffle=False),
    )

    if MODEL == "POINTER_GENERATOR":
        model = PointerGenerator(train_ds.vocab_size)
        for batch in train_loader:
            loss = model.train_one_batch(
                batch["input_ids"], batch["labels"], batch["input_lengths"]
            )
            print(loss)
    elif MODEL == "NEURAL_INTRA_ATTENTION":
        model = NeuralIntraAttention(train_ds.vocab_size, train_ds.tokenizer)
        for batch in train_loader:
            loss = model.train_one_batch(
                batch["input_ids"],
                batch["labels"],
                batch["oov_lists"],
                batch["input_lengths"],
            )
            print(loss)
    elif MODEL == "TRANSFORMER":
        model = TransformerSummarizer(train_ds.vocab_size, 128, 8, 2)
        # no end token, just like [56, 76, 34, 0, 0, 0, 0]
        for batch in train_loader:
            loss = model.train_one_batch(
                batch["input_ids"], batch["labels"], batch["input_lengths"]
            )
            print(loss)
