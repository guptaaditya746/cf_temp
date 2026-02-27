from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def apply_constraints_v2(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
    max_delta_per_step: Optional[float] = None,
    relational_linear: Optional[Dict[str, tuple[int, int, float]]] = None,
) -> tuple[np.ndarray, float, dict]:
    """
    Apply hard constraints and compute soft-violation diagnostics.

    relational_linear keys are arbitrary names, values are:
    (lhs_feature_idx, rhs_feature_idx, min_ratio)
    enforcing x[:, lhs] >= min_ratio * x[:, rhs] softly.
    """
    out = np.array(x_cf, copy=True, dtype=np.float64)
    x0 = np.asarray(x_original, dtype=np.float64)
    breakdown: dict[str, float] = {
        "immutable": 0.0,
        "bounds": 0.0,
        "max_delta_per_step": 0.0,
        "relational_linear": 0.0,
    }

    if bounds:
        for feature_idx, (min_val, max_val) in bounds.items():
            idx = int(feature_idx)
            lo = float(min_val)
            hi = float(max_val)
            before = np.array(out[:, idx], copy=True)
            out[:, idx] = np.clip(out[:, idx], lo, hi)
            breakdown["bounds"] += float(np.sum(np.abs(before - out[:, idx])))

    if immutable_features:
        for feature_idx in immutable_features:
            idx = int(feature_idx)
            diff = np.abs(out[:, idx] - x0[:, idx])
            breakdown["immutable"] += float(np.sum(diff))
            out[:, idx] = x0[:, idx]

    if max_delta_per_step is not None and out.shape[0] > 1:
        max_d = float(max_delta_per_step)
        if max_d > 0.0:
            d = np.diff(out, axis=0)
            over = np.abs(d) - max_d
            over = np.maximum(over, 0.0)
            breakdown["max_delta_per_step"] = float(np.sum(over))

    if relational_linear:
        rel_v = 0.0
        for _, (lhs_idx, rhs_idx, ratio) in relational_linear.items():
            li = int(lhs_idx)
            ri = int(rhs_idx)
            r = float(ratio)
            lhs = out[:, li]
            rhs = out[:, ri]
            rel_v += float(np.sum(np.maximum(0.0, r * rhs - lhs)))
        breakdown["relational_linear"] = rel_v

    total = float(sum(float(v) for v in breakdown.values()))
    return out, total, breakdown
