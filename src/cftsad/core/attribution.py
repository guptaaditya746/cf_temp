from __future__ import annotations

from typing import Any, Dict

import numpy as np


def attribution_from_delta(
    x: np.ndarray,
    x_cf: np.ndarray,
    top_k: int = 5,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    x_cf = np.asarray(x_cf, dtype=np.float64)
    delta = x_cf - x
    abs_delta = np.abs(delta)

    feature_importance = np.sum(abs_delta, axis=0)
    timestep_importance = np.sum(abs_delta, axis=1)
    edit_mask = abs_delta > float(eps)

    feat_order = np.argsort(-feature_importance)
    time_order = np.argsort(-timestep_importance)
    k = max(1, int(top_k))

    return {
        "feature_importance": feature_importance,
        "timestep_importance": timestep_importance,
        "feature_ranking": [int(i) for i in feat_order.tolist()],
        "timestep_ranking": [int(i) for i in time_order.tolist()],
        "top_features": [
            {"feature": int(i), "score": float(feature_importance[i])}
            for i in feat_order[:k]
        ],
        "top_timesteps": [
            {"timestep": int(i), "score": float(timestep_importance[i])}
            for i in time_order[:k]
        ],
        "edit_mask": edit_mask,
    }


def attribution_from_reconstruction_error(
    errors_per_timestep: np.ndarray,
    top_k: int = 5,
) -> Dict[str, Any]:
    err = np.asarray(errors_per_timestep, dtype=np.float64).reshape(-1)
    order = np.argsort(-err)
    k = max(1, int(top_k))
    return {
        "error_per_timestep": err,
        "error_timestep_ranking": [int(i) for i in order.tolist()],
        "top_error_timesteps": [
            {"timestep": int(i), "error": float(err[i])}
            for i in order[:k]
        ],
    }
