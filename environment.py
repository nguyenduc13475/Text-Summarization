import os
import sys

import matplotlib.pyplot as plt
from IPython.display import display


def detect_runtime_env():
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


def try_set_window_position(x=50, y=50):
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
        try:
            mgr.window.SetPosition((x, y))
            return
        except Exception:
            pass
    except Exception:
        pass


def adaptive_display(figure, ENV):
    match ENV:
        case "colab" | "notebook":
            display(figure)
        case "gui":
            plt.pause(0.01)
