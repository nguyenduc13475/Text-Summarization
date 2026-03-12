import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """
    Loads parameters from a YAML configuration file using OmegaConf.
    This enables dot-notation access (e.g., cfg.model.name) and strict typing.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        DictConfig: The configuration object.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"❌ Configuration file not found at: {path.resolve()}")

    try:
        config = OmegaConf.load(path)
        logger.info(f"✅ Successfully loaded config from {path}")
        return config
    except Exception as exc:
        logger.error(f"❌ Error parsing YAML file: {exc}")
        raise
