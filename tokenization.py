import json
import os
import re
from collections import Counter

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

CONTROL_TOKENS = {"<CAP>", "<ALLCAP>"}

from dataclasses import dataclass


@dataclass
class Encoding:
    tokens: list
    ids: list


class PointerGeneratorTokenizer:
    def __init__(self, vocab):
        if os.path.exists(vocab):
            with open(vocab, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
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

            with open("word_level_vocab.json", "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        self.id2token = {id: token for token, id in self.vocab.items()}

    def encode(self, text):
        tokens = []
        ids = []
        words = re.findall(r"\w+|[^\w\s]", text)
        for w in words:
            if w.isupper():
                tokens.append("<ALLCAP>")
                ids.append(self.vocab.get("<ALLCAP>"))
                lw = w.lower()
                tokens.append(lw)
                ids.append(self.vocab.get(lw, self.vocab.get("<unk>")))
            elif w[0].isupper() and w[1:].islower():
                tokens.append("<CAP>")
                ids.append(self.vocab.get("<CAP>"))
                lw = w.lower()
                tokens.append(lw)
                ids.append(self.vocab.get(lw, self.vocab.get("<unk>")))
            else:
                tokens.append(w)
                ids.append(self.vocab.get(w, self.vocab.get("<unk>")))

        return Encoding(tokens=tokens, ids=ids)

    def decode(self, ids):
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

    def get_vocab_size(self):
        return len(self.vocab)

    def token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<unk>"])

    def id_to_token(self, id):
        return self.id2token.get(id, "<unk>")


class TransformerTokenizer:
    def __init__(self, vocab, merges):
        if os.path.exists(vocab) and os.path.exists(merges):
            self.tokenizer = ByteLevelBPETokenizer(vocab, merges)
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

            self.tokenizer.save_model(".")

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, text):
        return self.tokenizer.decode(text)

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
