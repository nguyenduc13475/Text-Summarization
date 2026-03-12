from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import torch

from src.data.tokenization import PointerGeneratorTokenizer, TransformerTokenizer
from src.models.base_model import BaseSummarizationModel
from src.models.neural_intra_attention_model import NeuralIntraAttentionModel
from src.models.pointer_generator_network import PointerGeneratorNetwork
from src.models.transformer import Transformer

# Directories configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
VOCAB_DIR = BASE_DIR / "assets" / "vocab"
TRANSFORMER_VOCAB = VOCAB_DIR / "vocab.json"
TRANSFORMER_MERGES = VOCAB_DIR / "merges.txt"
POINTER_VOCAB = VOCAB_DIR / "word_level_vocab.json"

# Use a specific Union type for better clarity
AnyTokenizer = Union[PointerGeneratorTokenizer, TransformerTokenizer, None]


@unique
class ModelArchitecture(Enum):
    """Supported model architectures to prevent hard-coded string errors."""

    TRANSFORMER = "TRANSFORMER"
    POINTER_GENERATOR = "POINTER_GENERATOR_NETWORK"
    INTRA_ATTENTION = "NEURAL_INTRA_ATTENTION_MODEL"


# Model Registry to avoid long if-else chains
MODEL_REGISTRY: Dict[ModelArchitecture, Type[BaseSummarizationModel]] = {
    ModelArchitecture.TRANSFORMER: Transformer,
    ModelArchitecture.POINTER_GENERATOR: PointerGeneratorNetwork,
    ModelArchitecture.INTRA_ATTENTION: NeuralIntraAttentionModel,
}


def build_tokenizer(model_name: str) -> AnyTokenizer:
    """
    Factory to build the appropriate tokenizer for a given model.
    """
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)

    if model_name == "TRANSFORMER":
        return TransformerTokenizer(str(TRANSFORMER_VOCAB), str(TRANSFORMER_MERGES))
    elif model_name in ["POINTER_GENERATOR_NETWORK", "NEURAL_INTRA_ATTENTION_MODEL"]:
        return PointerGeneratorTokenizer(str(POINTER_VOCAB))
    return None


def build_model(
    model_arch: Union[str, ModelArchitecture],
    tokenizer: Any,
    device: Union[str, torch.device],
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseSummarizationModel:
    """
    Factory function to instantiate models.
    Supports both Enum and String for flexibility.
    """
    if isinstance(model_arch, str):
        model_arch = ModelArchitecture(model_arch)

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(f"Architecture '{model_arch}' not supported.")

    model_class = MODEL_REGISTRY[model_arch]

    # Extract model-specific parameters from config
    model_params = cfg.get("model", {}).get(model_arch.value, {}) if cfg else {}
    model_params.update(kwargs)

    return model_class(tokenizer=tokenizer, device=device, **model_params)
