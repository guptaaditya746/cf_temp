import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_counterfactual(
    x: np.ndarray,  # (L,F) original
    x_cf: np.ndarray,  # (L,F) counterfactual
    *,
    feature_names: Optional[List[str]] = None,
    edit_segment: Optional[Tuple[int, int]] = None,
    show_diff: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    General-purpose counterfactual plot for time-series windows.

    Args:
        x: original window (L,F)
        x_cf: counterfactual window (L,F)
        feature_names: optional list of feature names
        edit_segment: (start, end) tuple if segment substitution was used
        show_diff: whether to show |x - x_cf|
        title: plot title
    """
    assert x.shape == x_cf.shape, "x and x_cf must have same shape"
    L, F = x.shape

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    n_rows = F if not show_diff else 2 * F
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12, 2.2 * n_rows),
        sharex=True,
    )

    if n_rows == 1:
        axes = [axes]

    row = 0
    for f in range(F):
        ax = axes[row]
        ax.plot(x[:, f], label="original", linewidth=1.5)
        ax.plot(x_cf[:, f], label="counterfactual", linewidth=1.5, linestyle="--")

        if edit_segment is not None:
            ax.axvspan(
                edit_segment[0],
                edit_segment[1],
                color="red",
                alpha=0.15,
                label="edited segment" if f == 0 else None,
            )

        ax.set_ylabel(feature_names[f])
        ax.grid(True, alpha=0.3)

        if f == 0:
            ax.legend(loc="upper right")

        row += 1

        if show_diff:
            ax_diff = axes[row]
            diff = np.abs(x[:, f] - x_cf[:, f])
            ax_diff.plot(diff, color="black", linewidth=1.2)
            ax_diff.set_ylabel(f"|Δ {feature_names[f]}|")
            ax_diff.grid(True, alpha=0.3)
            row += 1

    axes[-1].set_xlabel("Time")

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout()
    # --- SAVE LOGIC ---
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        # Close to free memory (crucial if running in a loop or on a server)
        plt.close(fig)
    else:
        plt.show()
