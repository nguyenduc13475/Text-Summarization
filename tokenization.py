from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

if __name__ == "__main__":
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["train"]
    texts = [sample["article"] + " " + sample["highlights"] for sample in ds]

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(
        texts,
        vocab_size=50000,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
    )

    tokenizer.save_model(".")
