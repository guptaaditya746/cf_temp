from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.constraints import apply_constraints
from cftsad.core.distances import window_mse_distance
from cftsad.core.scoring import reconstruction_score
from cftsad.types import CFFailure, CFResult


def generate_nearest(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray,
    threshold: float,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    score_before = reconstruction_score(model, x)

    distances = np.asarray(
        [window_mse_distance(x, donor) for donor in normal_core],
        dtype=np.float64,
    )
    donor_idx = int(np.argmin(distances))
    donor_distance = float(distances[donor_idx])

    x_cf = apply_constraints(normal_core[donor_idx], x, immutable_features, bounds)
    score_after = reconstruction_score(model, x_cf)

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    if score_after <= threshold:
        return CFResult(
            x_cf=x_cf,
            score_cf=score_after,
            meta={
                "donor_idx": donor_idx,
                "donor_distance": donor_distance,
                "score_before": score_before,
                "score_after": score_after,
                "runtime_ms": runtime_ms,
            },
        )

    return CFFailure(
        reason="no_valid_cf",
        message="Nearest donor did not satisfy threshold.",
        diagnostics={
            "donor_idx": donor_idx,
            "donor_distance": donor_distance,
            "score_before": score_before,
            "score_after": score_after,
            "threshold": float(threshold),
            "runtime_ms": runtime_ms,
        },
    )
