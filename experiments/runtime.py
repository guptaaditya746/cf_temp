import os
import warnings
from datetime import datetime


def make_output_dir(base_dir="results"):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def detect_runtime():
    import torch

    if torch.cuda.is_available():
        return {
            "device": torch.device("cuda"),
            "accelerator": "cuda",
            "devices": 1,
        }
    if torch.backends.mps.is_available():
        return {
            "device": torch.device("mps"),
            "accelerator": "mps",
            "devices": 1,
        }
    return {
        "device": torch.device("cpu"),
        "accelerator": "cpu",
        "devices": 1,
    }


def configure_runtime():
    import matplotlib.pyplot as plt
    import seaborn as sns

    runtime = detect_runtime()
    device = runtime["device"]
    warnings.filterwarnings("ignore")
    plt.style.use("ggplot")
    sns.set_theme(style="whitegrid")
    print(f"Currently using: {device} ({runtime['accelerator']})")
    return device
