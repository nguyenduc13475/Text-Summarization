from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
from neural_intra_attention_model import NeuralIntraAttentionModel
from pointer_generator_network import PointerGeneratorNetwork
from tokenization import PointerGeneratorTokenizer
from transformer import Transformer
from utils import set_seed

set_seed()

# tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")
tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")

ds = CNNDailyMailDataset(
    split="train",
    tokenizer=tokenizer,
    dataset_length=None,
)
loader = DataLoader(
    ds,
    collate_fn=collate_fn,
    batch_sampler=DynamicBatchSampler(ds, max_tokens=10000, shuffle=True),
)

model = Transformer(
    tokenizer=tokenizer,
    d_model=128,
    nhead=8,
    num_layers=2,
    learning_rate=1e-3,
    device="cpu",
)

for batch_idx, batch in enumerate(loader):
    batch_output_ids = model.infer(
        batch["input_ids"],
        max_output_length=10,
        beam_width=4,
        return_embedding=True,
        return_attention=True,
    )["output_ids"]
    print(batch_output_ids)
