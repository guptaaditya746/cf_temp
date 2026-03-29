import os
import warnings
from datetime import datetime


def make_output_dir(base_dir="results"):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def configure_runtime():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    warnings.filterwarnings("ignore")
    plt.style.use("ggplot")
    sns.set_theme(style="whitegrid")
    print(f"Currently using: {device}")
    return device
