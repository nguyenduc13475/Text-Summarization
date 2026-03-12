import os
from pathlib import Path
from typing import Union

from src.models.factory import ModelArchitecture
from src.utils.environment import detect_runtime_env

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV = detect_runtime_env()


def get_workspace_dir() -> Path:
    """Automatically determine the working directory (Google Drive is preferred if running on Colab)."""
    if ENV == "colab" and os.path.exists("/content/drive/MyDrive"):
        drive_dir = Path("/content/drive/MyDrive/DATHAI")
        drive_dir.mkdir(parents=True, exist_ok=True)
        return drive_dir
    return BASE_DIR


def get_checkpoint_dir(model_name: Union[str, ModelArchitecture]) -> Path:
    """
    Get the checkpoint folder path.
    Handles both raw strings and ModelArchitecture Enums safely.
    """
    name_str = (
        model_name.value if isinstance(model_name, ModelArchitecture) else model_name
    )

    checkpoint_dir = get_workspace_dir() / f"{name_str.lower()}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_cache_dir() -> Path:
    """
    Retrieve or create a hidden directory for storing dataset and processing caches.
    Ensures the root directory remains clean.
    """
    cache_dir = get_workspace_dir() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
