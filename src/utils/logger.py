import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "Viettel_DSAI", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with dual-output (Console and File).
    All logs are timestamped and categorized by severity levels.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        log_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # File Outpu
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)

    return logger
