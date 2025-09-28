import torch


def tensor_dict_to_scalar(d):
    return {k: v.item() for k, v in d.items()}


def token_ids_to_text(tokenizer, ids, oov_list, vocab_size, return_output="text"):
    tokens = []
    for idx in ids:
        if idx < vocab_size:
            tok = tokenizer.id_to_token(idx)
        else:
            oov_idx = idx - vocab_size
            tok = oov_list[oov_idx]

        tokens.append(tok.replace("Ġ", ""))

    if return_output == "text":
        return "".join(tokens).strip()
    elif return_output == "list":
        return tokens
    else:
        return ["".join(tokens).strip(), tokens]


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
