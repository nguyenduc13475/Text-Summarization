import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

CONTROL_TOKENS = {"<CAP>", "<ALLCAP>"}


@dataclass
class Encoding:
    tokens: List[str]
    ids: List[int]


class PointerGeneratorTokenizer:
    def __init__(self, vocab: str) -> None:
        if os.path.exists(vocab):
            with open(vocab, "r", encoding="utf-8") as f:
                self.vocab: Dict[str, int] = json.load(f)
        else:
            counter = Counter()
            ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["train"]
            print("Building tokenizer...")
            for i, sample in enumerate(ds):
                if i % 1000 == 0:
                    print(f"{i + 1}/{len(ds)} samples")
                text = sample["article"] + " " + sample["highlights"]
                words = re.findall(r"\w+|[^\w\s]", text.lower())
                counter.update(words)
            self.vocab = {
                "<pad>": 0,
                "<unk>": 1,
                "<s>": 2,
                "</s>": 3,
                "<CAP>": 4,
                "<ALLCAP>": 5,
            }
            for word, _ in counter.most_common(50000 - len(self.vocab)):
                self.vocab[word] = len(self.vocab)

            with open("assets/vocab/word_level_vocab.json", "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        self.id2token: Dict[int, str] = {id: token for token, id in self.vocab.items()}

    def encode(self, text: str) -> Encoding:
        tokens: List[str] = []
        ids: List[int] = []
        words = re.findall(r"\w+|[^\w\s]", text)
        for w in words:
            if w.isupper():
                tokens.append("<ALLCAP>")
                ids.append(self.vocab.get("<ALLCAP>", 1))
                lw = w.lower()
                tokens.append(lw)
                ids.append(self.vocab.get(lw, self.vocab.get("<unk>", 1)))
            elif w[0].isupper() and w[1:].islower():
                tokens.append("<CAP>")
                ids.append(self.vocab.get("<CAP>", 1))
                lw = w.lower()
                tokens.append(lw)
                ids.append(self.vocab.get(lw, self.vocab.get("<unk>", 1)))
            else:
                tokens.append(w)
                ids.append(self.vocab.get(w, self.vocab.get("<unk>", 1)))

        return Encoding(tokens=tokens, ids=ids)

    def decode(self, ids: List[int]) -> str:
        result = ""
        i = 0

        while i < len(ids):
            tok_id = ids[i]
            tok = self.id2token.get(tok_id, "<unk>")

            if tok == "</s>" or tok == "<pad>":
                return result
            if tok == "<CAP>" and i + 1 < len(ids):
                next_tok = self.id2token.get(ids[i + 1], "<unk>")
                word = next_tok.capitalize()
                result += (
                    " "
                    if result
                    and (
                        result[-1].isalnum() or result[-1] in {")", "]", "}", ">", "-"}
                    )
                    else ""
                ) + word
                i += 2
            elif tok == "<ALLCAP>" and i + 1 < len(ids):
                next_tok = self.id2token.get(ids[i + 1], "<unk>")
                word = next_tok.upper()
                result += (
                    " "
                    if result
                    and (
                        result[-1].isalnum() or result[-1] in {")", "]", "}", ">", "-"}
                    )
                    else ""
                ) + word
                i += 2
            else:
                if tok in {".", ",", "!", "?", ";", ":"}:
                    result = result.rstrip() + tok + " "
                elif tok in {"(", "[", "{"}:
                    result += (" " if result and result[-1].isalnum() else "") + tok
                elif tok in {")", "]", "}"}:
                    result = result.rstrip() + tok
                else:
                    result += (
                        " "
                        if result
                        and (
                            result[-1].isalnum()
                            or (
                                (tok[0].isalnum() or tok[0] == "<")
                                and result[-1] in {")", "]", "}", ">", "-"}
                            )
                        )
                        else ""
                    ) + tok
                i += 1

        return result.strip()

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["<unk>"])

    def id_to_token(self, id: int) -> str:
        return self.id2token.get(id, "<unk>")


class TransformerTokenizer:
    def __init__(self, vocab: str, merges: str) -> None:
        if os.path.exists(vocab) and os.path.exists(merges):
            self.tokenizer = ByteLevelBPETokenizer(vocab, merges)
            self.tokenizer.add_special_tokens(["<pad>", "<unk>", "<s>", "</s>"])
        else:
            ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["train"]
            texts = [sample["article"] + " " + sample["highlights"] for sample in ds]
            self.tokenizer = ByteLevelBPETokenizer()

            self.tokenizer.train_from_iterator(
                texts,
                vocab_size=50000,
                min_frequency=2,
                special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
            )

            os.makedirs("assets/vocab", exist_ok=True)
            self.tokenizer.save_model("assets/vocab")

    def encode(self, text: str) -> Any:
        return self.tokenizer.encode(text)

    def decode(self, text: List[int]) -> str:
        return self.tokenizer.decode(text, skip_special_tokens=True)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str:
        token = self.tokenizer._tokenizer.id_to_token(id)
        return token if token is not None else "<unk>"

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
