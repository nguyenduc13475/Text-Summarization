import os
import pickle
import random

import numpy as np
import torch

name_to_latex = {
    "nll_loss": r"$L_{\text{nll}}$",
    "cov_loss": r"$L_{\text{cov}}$",
    "rl_loss": r"$L_{\text{rl}}$",
    "total_loss": r"$L$",
}


def tensor_dict_to_scalar(d):
    return {k: v.item() for k, v in d.items()}


def token_ids_to_text(tokenizer, ids, oov_list, return_output="text"):
    vocab_size = tokenizer.get_vocab_size()
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if return_output == "text":
        oov_tokens = []
        for idx in ids:
            if idx >= vocab_size:
                oov_idx = idx - vocab_size
                oov_token = oov_list[oov_idx]
                oov_tokens.append(oov_token)

        sentence = tokenizer.decode(ids)
        for token in oov_tokens:
            sentence = sentence.replace("<unk>", token, 1)
        return sentence
    elif return_output == "list":
        tokens = []
        for idx in ids:
            if idx < vocab_size:
                tokens.append(tokenizer.id_to_token(idx))
                if tokens[-1] == "</s>":
                    break
            else:
                tokens.append(oov_list[idx - vocab_size])

        return tokens
    else:
        oov_tokens = []
        tokens = []
        for idx in ids:
            if idx < vocab_size:
                tokens.append(tokenizer.id_to_token(idx))
                if tokens[-1] == "</s>":
                    break
            else:
                oov_idx = idx - vocab_size
                oov_token = oov_list[oov_idx]
                oov_tokens.append(oov_token)
                tokens.append(oov_token)

        sentence = tokenizer.decode(ids)
        for token in oov_tokens:
            sentence = sentence.replace("<unk>", token, 1)
        return sentence, tokens


def text_to_token_ids(tokenizer, input_text, oov_list=[]):
    vocab_size = tokenizer.get_vocab_size()
    encoded_text = tokenizer.encode(input_text)

    ids = []

    for token_idx, token in zip(encoded_text.ids, encoded_text.tokens):
        if token_idx == tokenizer.token_to_id("<unk>"):
            try:
                ids.append(vocab_size + oov_list.index(token))
            except:
                ids.append(vocab_size + len(oov_list))
                oov_list.append(token)

        else:
            ids.append(token_idx)
    return ids


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_log_file(log, log_file):
    print(log)
    log_file.write(log + "\n")


def save_checkpoint(model, checkpoint_file, save_optimizer=True):
    if save_optimizer:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_file)
    else:
        torch.save(model.state_dict(), checkpoint_file)


def load_checkpoint(model, checkpoint_file, map_location="cpu"):
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def cache(func, name):
    cache_file = f"cache/{name}"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        os.makedirs("cache", exist_ok=True)
        result = func()
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result


def create_appearance_boost(batch_input_ids, model, original_attention, num_oovs=0):
    batch_size = batch_input_ids.shape[0]
    appearance_boost = torch.zeros(
        batch_size, model.vocab_size + num_oovs, device=model.device
    )
    for i in range(batch_size):
        appearance_boost[i, torch.unique(batch_input_ids[i])] = original_attention
    appearance_boost = appearance_boost[:, model.end_token :]
    return appearance_boost
