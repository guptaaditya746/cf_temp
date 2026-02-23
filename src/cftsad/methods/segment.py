from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.constraints import apply_constraints
from cftsad.core.distances import window_mse_distance
from cftsad.core.scoring import reconstruction_errors_per_timestep, reconstruction_score
from cftsad.types import CFFailure, CFResult


def _largest_contiguous_region(indices: np.ndarray) -> tuple[int, int] | None:
    if indices.size == 0:
        return None
    sorted_idx = np.sort(indices)
    best_start = int(sorted_idx[0])
    best_end = int(sorted_idx[0])
    cur_start = int(sorted_idx[0])
    cur_end = int(sorted_idx[0])

    for idx in sorted_idx[1:]:
        val = int(idx)
        if val == cur_end + 1:
            cur_end = val
        else:
            if (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start, cur_end = val, val

    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end

    return best_start, best_end


def detect_anomalous_segment(model: object, x: np.ndarray) -> tuple[int, int] | None:
    """
    Mandatory v1 segment detection:
    1) per-timestep reconstruction error
    2) top 10% timesteps
    3) largest contiguous region
    4) +/-2 padding
    5) min length = max(5, L//20)
    Returns inclusive [start, end].
    """
    L = x.shape[0]
    if L == 0:
        return None

    errors = reconstruction_errors_per_timestep(model, x)
    n_top = max(1, int(np.ceil(0.1 * L)))
    top_idx = np.argpartition(errors, -n_top)[-n_top:]

    region = _largest_contiguous_region(top_idx)
    if region is None:
        return None

    start, end = region
    start = max(0, start - 2)
    end = min(L - 1, end + 2)

    min_len = max(5, L // 20)
    cur_len = end - start + 1
    if cur_len < min_len:
        grow = min_len - cur_len
        left = grow // 2
        right = grow - left
        start = max(0, start - left)
        end = min(L - 1, end + right)
        while (end - start + 1) < min_len and start > 0:
            start -= 1
        while (end - start + 1) < min_len and end < (L - 1):
            end += 1

    return int(start), int(end)


def _smooth_boundaries(x_cf: np.ndarray, x_original: np.ndarray, start: int, end: int) -> np.ndarray:
    out = np.array(x_cf, copy=True)
    if start > 0:
        out[start] = 0.5 * x_original[start - 1] + 0.5 * out[start]
    if end < x_cf.shape[0] - 1:
        out[end] = 0.5 * out[end] + 0.5 * x_original[end + 1]
    return out


def generate_segment(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray,
    threshold: float,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
    smoothing: bool = False,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    score_before = reconstruction_score(model, x)

    segment = detect_anomalous_segment(model, x)
    if segment is None:
        return CFFailure(
            reason="segment_detection_failed",
            message="Unable to detect anomalous segment.",
            diagnostics={"threshold": float(threshold), "score_before": score_before},
        )

    start, end = segment
    distances = np.asarray(
        [window_mse_distance(x, donor) for donor in normal_core],
        dtype=np.float64,
    )
    donor_idx = int(np.argmin(distances))

    x_cf = np.array(x, copy=True)
    x_cf[start : end + 1] = normal_core[donor_idx, start : end + 1]
    if smoothing:
        x_cf = _smooth_boundaries(x_cf, x, start, end)

    x_cf = apply_constraints(x_cf, x, immutable_features, bounds)
    score_after = reconstruction_score(model, x_cf)

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    if score_after <= threshold:
        return CFResult(
            x_cf=x_cf,
            score_cf=score_after,
            meta={
                "segment_start": int(start),
                "segment_end": int(end),
                "donor_idx": donor_idx,
                "score_before": score_before,
                "score_after": score_after,
                "smoothing_used": bool(smoothing),
                "runtime_ms": runtime_ms,
            },
        )

    return CFFailure(
        reason="no_valid_cf",
        message="Segment substitution candidate did not satisfy threshold.",
        diagnostics={
            "segment_start": int(start),
            "segment_end": int(end),
            "donor_idx": donor_idx,
            "score_before": score_before,
            "score_after": score_after,
            "threshold": float(threshold),
            "smoothing_used": bool(smoothing),
            "runtime_ms": runtime_ms,
        },
    )
