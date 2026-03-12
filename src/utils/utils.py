import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.utils.paths import get_cache_dir

name_to_latex: Dict[str, str] = {
    "nll_loss": r"$L_{\text{nll}}$",
    "cov_loss": r"$L_{\text{cov}}$",
    "rl_loss": r"$L_{\text{rl}}$",
    "total_loss": r"$L$",
}


def tensor_dict_to_scalar(d: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: v.item() for k, v in d.items()}


def token_ids_to_text(
    tokenizer: Any,
    ids: Union[torch.Tensor, List[int]],
    oov_list: List[str],
    return_output: str = "text",
) -> Union[str, List[str], Tuple[str, List[str]]]:
    vocab_size = tokenizer.get_vocab_size()
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    if return_output == "text":
        oov_tokens: List[str] = []
        for idx in ids:
            if idx >= vocab_size:
                oov_idx = idx - vocab_size
                oov_token = oov_list[oov_idx]
                oov_tokens.append(oov_token)

        sentence: str = tokenizer.decode(ids)
        for token in oov_tokens:
            sentence = sentence.replace("<unk>", token, 1)
        return sentence

    elif return_output == "list":
        tokens: List[str] = []
        for idx in ids:
            if idx < vocab_size:
                tokens.append(tokenizer.id_to_token(idx))
                if tokens[-1] == "</s>":
                    break
            else:
                tokens.append(oov_list[idx - vocab_size])
        return tokens

    else:
        oov_tokens: List[str] = []
        tokens: List[str] = []
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

        sentence: str = tokenizer.decode(ids)
        for token in oov_tokens:
            sentence = sentence.replace("<unk>", token, 1)
        return sentence, tokens


def text_to_token_ids(
    tokenizer: Any, input_text: str, oov_list: Optional[List[str]] = None
) -> List[int]:
    if oov_list is None:
        oov_list = []
    vocab_size = tokenizer.get_vocab_size()
    encoded_text = tokenizer.encode(input_text)
    ids: List[int] = []

    for token_idx, token in zip(encoded_text.ids, encoded_text.tokens):
        if token_idx == tokenizer.token_to_id("<unk>"):
            try:
                ids.append(vocab_size + oov_list.index(token))
            except ValueError:
                ids.append(vocab_size + len(oov_list))
                oov_list.append(token)
        else:
            ids.append(token_idx)

    return ids


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_log_file(log: str, log_file: Any) -> None:
    print(log)
    log_file.write(log + "\n")


def save_checkpoint(
    model: Any, checkpoint_file: Union[str, Path], save_optimizer: bool = True
) -> None:
    if save_optimizer:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_file)
    else:
        torch.save(model.state_dict(), checkpoint_file)


def load_checkpoint(
    model: Any,
    checkpoint_file: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> None:
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if hasattr(model, "optimizer") and "optimizer_state_dict" in checkpoint:
            model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def cache(func: Callable[[], Any], name: str) -> Any:
    cache_dir = get_cache_dir()
    cache_file = cache_dir / name

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        result = func()
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result


def create_appearance_boost(
    batch_input_ids: torch.Tensor,
    model: Any,
    original_attention: float,
    num_oovs: int = 0,
) -> torch.Tensor:
    batch_size = batch_input_ids.shape[0]
    appearance_boost = torch.zeros(
        batch_size, model.vocab_size + num_oovs, device=model.device
    )
    for i in range(batch_size):
        appearance_boost[i, torch.unique(batch_input_ids[i])] = original_attention
    appearance_boost = appearance_boost[:, model.end_token :]
    return appearance_boost


def find_latest_checkpoint(
    checkpoint_folder: Union[str, Path],
) -> Tuple[Optional[str], Optional[int]]:
    folder_path = str(checkpoint_folder)
    if os.path.exists(folder_path) and any(
        re.match(r"^checkpoint_([0-9]\d*)\.pt$", f) for f in os.listdir(folder_path)
    ):
        latest_checkpoint = max(
            int(m.group(1))
            for f in os.listdir(folder_path)
            if (m := re.match(r"^checkpoint_([0-9]\d*)\.pt$", f))
        )
        return f"{folder_path}/checkpoint_{latest_checkpoint}.pt", latest_checkpoint
    return None, None
