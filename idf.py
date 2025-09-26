import json
import math
from collections import defaultdict

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


def compute_idf(texts, tokenizer):
    df_counts = defaultdict(int)
    N = len(texts)

    for text in texts:
        tokens = tokenizer.encode(text).ids
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df_counts[token] += 1

    idf = {}
    for token, df in df_counts.items():
        idf[token] = math.log(N / (df + 1))
    return idf


if __name__ == "__main__":
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["train"]
    texts = [sample["article"] + " " + sample["highlights"] for sample in ds]

    tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    idf_raw = compute_idf(texts, tokenizer)

    vocab = tokenizer.get_vocab()
    vocab_inverse = {idx: token for token, idx in vocab.items()}

    idf_with_tokens = {
        vocab_inverse.get(token_idx, f"<unk_{token_idx}>"): val
        for token_idx, val in idf_raw.items()
    }

    with open("idf.json", "w", encoding="utf-8") as f:
        json.dump(idf_with_tokens, f, ensure_ascii=False, indent=2)
