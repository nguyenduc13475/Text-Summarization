import abc
from contextlib import nullcontext
from typing import Any, Dict, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.utils import tensor_dict_to_scalar


class ModelOutput(TypedDict):
    """
    Standardized output structure for all summarization models.
    Ensures consistency across Training, Inference, and Evaluation pipelines.
    """

    total_loss: Optional[torch.Tensor]
    nll_loss: Optional[torch.Tensor]
    rl_loss: Optional[torch.Tensor]
    output_ids: Optional[torch.Tensor]
    cross_attention_distributions: Optional[torch.Tensor]  # For interpretability
    input_embeddings: Optional[torch.Tensor]  # For t-SNE visualization


class BaseSummarizationModel(nn.Module, abc.ABC):
    """
    Abstract Base Class for all Text Summarization architectures.
    Follows the Interface Segregation Principle.
    """

    def __init__(self, device: Union[str, torch.device]):
        super().__init__()
        self.device = torch.device(device)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters. Useful for model card reporting."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _safe_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Safely handle Out-Of-Vocabulary (OOV) token IDs by mapping them to the <unk> token ID.
        Shared across all models dealing with dynamic pointer networks.
        """
        if not hasattr(self, "vocab_size") or not hasattr(self, "unknown_token"):
            return ids

        return torch.where(
            ids >= self.vocab_size,
            torch.tensor(self.unknown_token, device=self.device),
            ids,
        )

    def _safe_embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Apply safe ID mapping before passing through the embedding layer."""
        if not hasattr(self, "embedding_layer"):
            raise NotImplementedError(
                "Model must define self.embedding_layer to use _safe_embed"
            )

        safe_ids = self._safe_ids(ids)
        return self.embedding_layer(safe_ids)

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Standard forward pass for PyTorch compatibility."""
        pass

    @abc.abstractmethod
    def compute_loss(
        self, batch: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate loss components from a data batch.
        """
        pass

    @abc.abstractmethod
    def infer(self, batch_input_ids: torch.Tensor, **kwargs: Any) -> ModelOutput:
        """
        Generate summary sequences for a given input.
        """
        pass

    def setup_training_env(
        self, learning_rate: float, loss_scale: float = 1.0, weight_decay: float = 1e-5
    ) -> None:
        """
        Standardize optimizer and AMP scaler initialization.
        Ensures all models use consistent optimization configurations.
        """
        self.loss_scale = loss_scale
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

    def train_one_batch(self, **kwargs: Any) -> Dict[str, float]:
        """
        Generic training loop with Automatic Mixed Precision (AMP).
        Models with standard single-pass loss (Transformer, PGN) inherit this directly.
        """
        self.train()
        self.optimizer.zero_grad()

        is_cuda = self.device.type == "cuda"
        with torch.amp.autocast(device_type="cuda") if is_cuda else nullcontext():
            losses = self.compute_loss(**kwargs)
            loss_to_optimize = (
                losses.get("total_loss", sum(losses.values())) * self.loss_scale
            )

        if is_cuda and self.scaler is not None:
            self.scaler.scale(loss_to_optimize).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_to_optimize.backward()
            self.optimizer.step()

        return tensor_dict_to_scalar(losses)

    def validate_one_batch(self, **kwargs: Any) -> Dict[str, float]:
        """Generic validation loop."""
        self.eval()
        with torch.no_grad():
            losses = self.compute_loss(**kwargs)
        return tensor_dict_to_scalar(losses)
