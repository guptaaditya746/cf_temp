from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def apply_constraints(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Apply v1 constraints: optional bounds clipping and immutable features."""
    out = np.array(x_cf, copy=True)

    if bounds:
        for feature_idx, (min_val, max_val) in bounds.items():
            out[:, int(feature_idx)] = np.clip(
                out[:, int(feature_idx)],
                float(min_val),
                float(max_val),
            )

    if immutable_features:
        for feature_idx in immutable_features:
            idx = int(feature_idx)
            out[:, idx] = x_original[:, idx]

    return out
