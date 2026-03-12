import os
import sys
from typing import Literal

import matplotlib.pyplot as plt
import torch
from IPython.display import display

RuntimeEnv = Literal["colab", "notebook", "gui", "headless"]


def detect_runtime_env() -> RuntimeEnv:
    """Detects the current execution environment."""
    if "google.colab" in sys.modules:
        return "colab"
    if "ipykernel" in sys.modules:
        return "notebook"
    if (
        sys.platform.startswith("win")
        or sys.platform.startswith("darwin")
        or os.getenv("DISPLAY")
    ):
        return "gui"
    return "headless"


def get_device_report() -> str:
    """Generates a hardware diagnostic report for the logs."""
    from src.utils.hpc_utils import get_hpc_capability_report

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hpc_info = get_hpc_capability_report()

    lines = [
        f"Execution Provider: {device.upper()}",
        f"PyTorch Version: {torch.__version__}",
    ]

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f"Primary Accelerator: {gpu_name}")
        lines.append(f"Total VRAM: {vram_gb:.2f} GB")
        lines.append(f"Compute Capability: {hpc_info.get('compute_capability')}")
        lines.append(
            f"Hardware Support: BF16={hpc_info['bf16_supported']}, TF32={hpc_info['tf32_supported']}"
        )

    return " | ".join(lines)


def try_set_window_position(x: int = 50, y: int = 50) -> None:
    """Set the initial position of the plot window if using a GUI backend."""
    try:
        mgr = plt.get_current_fig_manager()
        try:
            mgr.window.wm_geometry(f"+{x}+{y}")
            return
        except Exception:
            pass
        try:
            mgr.window.move(x, y)
            return
        except Exception:
            pass
    except Exception:
        pass


def adaptive_display(figure: plt.Figure, ENV: str) -> None:
    """Render plots based on the detected environment."""
    if ENV in ("colab", "notebook"):
        display(figure)
    elif ENV == "gui":
        plt.pause(0.01)
