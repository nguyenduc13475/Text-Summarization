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


def token_ids_to_text(tokenizer, ids, oov_list, vocab_size, return_output="text"):
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
            else:
                tokens.append(oov_list[idx - vocab_size])

        return tokens
    else:
        oov_tokens = []
        tokens = []
        for idx in ids:
            if idx < vocab_size:
                tokens.append(tokenizer.id_to_token(idx))
            else:
                oov_idx = idx - vocab_size
                oov_token = oov_list[oov_idx]
                oov_tokens.append(oov_token)
                tokens.append(oov_token)

        sentence = tokenizer.decode(ids)
        for token in oov_tokens:
            sentence = sentence.replace("<unk>", token, 1)
        return sentence, tokens


def pad_and_stack(tensor_list, pad_value=0.0):
    max_len = max(t.size(0) for t in tensor_list)
    padded = []
    for t in tensor_list:
        pad_size = max_len - t.size(0)
        if pad_size > 0:
            padded_t = torch.cat(
                [t, torch.full((pad_size,), pad_value, device=t.device)]
            )
        else:
            padded_t = t
        padded.append(padded_t)
    return torch.stack(padded, dim=0)


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
