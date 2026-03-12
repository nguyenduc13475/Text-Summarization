import logging
from typing import Any, Dict

import torch

logger = logging.getLogger("Viettel_DSAI")


def optimize_cuda_performance() -> None:
    """
    Optimizes CUDA performance settings for high-end GPUs (e.g., NVIDIA A100, B200).
    Enables TensorFloat-32 (TF32) and CuDNN auto-tuner.
    """
    if torch.cuda.is_available():
        # Enable TF32 for matmul and convolutions
        # This increases throughput with minimal impact on precision.
        torch.set_float32_matmul_precision("high")

        # Benchmarking helps CuDNN find the best algorithm for the current hardware
        torch.backends.cudnn.benchmark = True

        # Disabling deterministic mode for training speed boost unless reproducibility is strictly required
        torch.backends.cudnn.deterministic = False

        logger.info(
            "HPC: CUDA performance optimizations applied (TF32 enabled, CuDNN Benchmark activated)."
        )


def get_hpc_capability_report() -> Dict[str, Any]:
    """
    Analyzes hardware capabilities and returns a technical report.
    """
    report = {
        "device_count": torch.cuda.device_count(),
        "bf16_supported": False,
        "tf32_supported": False,
    }

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        report["compute_capability"] = f"{major}.{minor}"

        # BF16 is preferred for training on Ampere (A100) or newer (B200)
        if major >= 8:
            report["bf16_supported"] = True
            report["tf32_supported"] = True

    return report
