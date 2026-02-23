from __future__ import annotations

import numpy as np


def window_mse_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared distance between two windows with shape (L, F)."""
    diff = x - y
    return float(np.mean(diff * diff))
