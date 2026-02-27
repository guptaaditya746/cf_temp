from __future__ import annotations

from typing import Any, Dict

import numpy as np


def build_explainability_meta(
    x: np.ndarray,
    x_cf: np.ndarray,
    top_k: int = 3,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    x_cf = np.asarray(x_cf, dtype=np.float64)
    delta = x_cf - x
    abs_delta = np.abs(delta)

    feature_change = np.sum(abs_delta, axis=0)
    timestep_change = np.sum(abs_delta, axis=1)
    edit_mask = abs_delta > float(eps)

    feature_order = np.argsort(-feature_change)
    timestep_order = np.argsort(-timestep_change)
    k = max(1, int(top_k))

    return {
        "changed_features": [int(i) for i in feature_order.tolist()],
        "changed_timesteps": [int(i) for i in timestep_order.tolist()],
        "top_changed_features": [
            {"feature": int(i), "abs_delta_sum": float(feature_change[i])}
            for i in feature_order[:k]
        ],
        "top_changed_timesteps": [
            {"timestep": int(i), "abs_delta_sum": float(timestep_change[i])}
            for i in timestep_order[:k]
        ],
        "edit_mask": edit_mask,
        "delta_l1": float(np.sum(abs_delta)),
        "delta_l2": float(np.sqrt(np.sum(delta * delta))),
    }
