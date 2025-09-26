from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from neural_intra_attention import NeuralIntraAttention
from pointer_generator import PointerGenerator

MODEL = "NEURAL_INTRA_ATTENTION"

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
        model = NeuralIntraAttention(train_ds.vocab_size)
        for batch in train_loader:
            outputs = model.infer(batch["input_ids"][0], 10, 3)
            print(outputs.shape)
