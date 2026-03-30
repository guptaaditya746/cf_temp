from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.candidates import compute_candidate_metrics, rank_candidates
from cftsad.core.constraints import apply_constraints
from cftsad.core.constraints_v2 import apply_constraints_v2
from cftsad.core.postprocess import (
    build_candidate_summary,
    build_explainability_meta,
    build_score_summary,
)
from cftsad.core.distances import window_mse_distance
from cftsad.core.scoring import reconstruction_errors_per_timestep, reconstruction_score
from cftsad.types import CFFailure, CFResult


def _score_candidate(
    model: object,
    x_cf: np.ndarray,
    score_fn: Optional[Callable[[np.ndarray], float]],
) -> float:
    if score_fn is not None:
        return float(score_fn(np.asarray(x_cf, dtype=np.float64)))
    return reconstruction_score(model, x_cf)


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


def detect_candidate_segments(
    model: object,
    x: np.ndarray,
    n_segments: int = 4,
) -> list[tuple[int, int]]:
    L = x.shape[0]
    if L == 0:
        return []

    errors = reconstruction_errors_per_timestep(model, x)
    base = detect_anomalous_segment(model, x)
    if base is None:
        return []

    segments = {base}
    center = int(np.argmax(errors))
    lengths = sorted(set([max(5, L // 40), max(5, L // 20), max(5, L // 10)]))

    for length in lengths:
        half = length // 2
        start = max(0, center - half)
        end = min(L - 1, start + length - 1)
        start = max(0, end - length + 1)
        segments.add((int(start), int(end)))

    ranked = sorted(
        list(segments),
        key=lambda se: -float(np.mean(errors[se[0] : se[1] + 1])),
    )
    return ranked[: max(1, int(n_segments))]


def _crossfade_boundaries(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    start: int,
    end: int,
    width: int = 3,
) -> np.ndarray:
    out = np.array(x_cf, copy=True)
    w = max(1, int(width))

    for i in range(1, w + 1):
        li = start - i
        if li >= 0:
            alpha = float(i) / float(w + 1)
            out[li] = (1.0 - alpha) * x_original[li] + alpha * out[li]
        ri = end + i
        if ri < out.shape[0]:
            alpha = float(i) / float(w + 1)
            out[ri] = (1.0 - alpha) * x_original[ri] + alpha * out[ri]
    return out


def generate_segment(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray,
    threshold: float,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
    smoothing: bool = False,
    n_segments: int = 4,
    top_k_donors: int = 8,
    context_width: int = 2,
    crossfade_width: int = 3,
    use_constraints_v2: bool = False,
    max_delta_per_step: float | None = None,
    relational_linear: Dict[str, tuple[int, int, float]] | None = None,
    score_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    score_before = _score_candidate(model, x, score_fn)
    thr = float(threshold)
    errors = reconstruction_errors_per_timestep(model, x)

    segments = detect_candidate_segments(model, x, n_segments=n_segments)
    if not segments:
        return CFFailure(
            reason="segment_detection_failed",
            message="Unable to detect anomalous segment.",
            diagnostics={"threshold": thr, "score_before": score_before},
        )

    candidates = []
    attempted = 0
    best_constraint_violation = np.inf
    best_constraint_breakdown = None
    segment_search_summary = []

    for start, end in segments:
        c0 = max(0, start - int(context_width))
        c1 = min(x.shape[0] - 1, end + int(context_width))
        seg_x = x[c0 : c1 + 1]
        donor_dists = np.asarray(
            [window_mse_distance(seg_x, donor[c0 : c1 + 1]) for donor in normal_core],
            dtype=np.float64,
        )
        k = max(1, min(int(top_k_donors), int(donor_dists.shape[0])))
        top_idx = np.argpartition(donor_dists, k - 1)[:k]
        top_idx = top_idx[np.argsort(donor_dists[top_idx])]
        segment_search_summary.append(
            {
                "segment_start": int(start),
                "segment_end": int(end),
                "segment_length": int(end - start + 1),
                "touches_left_boundary": bool(start == 0),
                "touches_right_boundary": bool(end == x.shape[0] - 1),
                "context_start": int(c0),
                "context_end": int(c1),
                "segment_mean_error": float(np.mean(errors[start : end + 1])),
                "segment_max_error": float(np.max(errors[start : end + 1])),
                "best_donor_distance": float(np.min(donor_dists)),
                "median_donor_distance": float(np.median(donor_dists)),
                "top_donors": [
                    {
                        "donor_idx": int(donor_idx),
                        "distance": float(donor_dists[int(donor_idx)]),
                    }
                    for donor_idx in top_idx[: min(3, len(top_idx))].tolist()
                ],
            }
        )

        for donor_idx in top_idx:
            attempted += 1
            x_cf = np.array(x, copy=True)
            x_cf[start : end + 1] = normal_core[int(donor_idx), start : end + 1]
            if smoothing:
                x_cf = _crossfade_boundaries(
                    x_cf=x_cf,
                    x_original=x,
                    start=start,
                    end=end,
                    width=int(crossfade_width),
                )
            if use_constraints_v2:
                x_cf, c_v, c_break = apply_constraints_v2(
                    x_cf,
                    x,
                    immutable_features=immutable_features,
                    bounds=bounds,
                    max_delta_per_step=max_delta_per_step,
                    relational_linear=relational_linear,
                )
            else:
                x_cf = apply_constraints(x_cf, x, immutable_features, bounds)
                c_v, c_break = 0.0, None
            if c_v < best_constraint_violation:
                best_constraint_violation = float(c_v)
                best_constraint_breakdown = c_break
            score_after = _score_candidate(model, x_cf, score_fn)
            cand = compute_candidate_metrics(
                x=x,
                x_cf=x_cf,
                score_cf=score_after,
                normal_core=normal_core,
            )
            cand.diagnostics = {
                "segment_start": int(start),
                "segment_end": int(end),
                "context_start": int(c0),
                "context_end": int(c1),
                "donor_idx": int(donor_idx),
                "segment_distance": float(donor_dists[int(donor_idx)]),
                "constraint_violation": float(c_v),
                "constraint_breakdown": c_break,
            }
            candidates.append(cand)

    ranked = rank_candidates(candidates, threshold=thr)
    best = ranked[0] if ranked else None
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    common_meta = {
        "segment_candidates": segment_search_summary,
        "localizer_peak_timestep": int(np.argmax(errors)),
        "localizer_peak_error": float(np.max(errors)),
        "smoothing_used": bool(smoothing),
        "n_segments_considered": int(len(segments)),
        "n_candidates_evaluated": int(attempted),
        "best_candidate_summary": build_candidate_summary(x, best, threshold=thr),
        "runtime_ms": runtime_ms,
        "score_source": "custom" if score_fn is not None else "reconstruction_score",
    }
    if best is not None and best.score_cf <= thr:
        meta = {
            "segment_start": int(best.diagnostics["segment_start"]),
            "segment_end": int(best.diagnostics["segment_end"]),
            "context_start": int(best.diagnostics["context_start"]),
            "context_end": int(best.diagnostics["context_end"]),
            "donor_idx": int(best.diagnostics["donor_idx"]),
            "segment_distance": float(best.diagnostics["segment_distance"]),
            "constraint_violation": float(best.diagnostics["constraint_violation"]),
            "constraint_breakdown": best.diagnostics["constraint_breakdown"],
        }
        meta.update(build_score_summary(score_before, float(best.score_cf), thr))
        meta.update(common_meta)
        meta.update(build_explainability_meta(x, best.x_cf))
        return CFResult(x_cf=best.x_cf, score_cf=float(best.score_cf), meta=meta)

    return CFFailure(
        reason="no_valid_cf",
        message="Segment substitution candidate did not satisfy threshold.",
        diagnostics={
            **build_score_summary(
                score_before,
                None if best is None else float(best.score_cf),
                thr,
            ),
            **common_meta,
            "best_constraint_violation": (
                None if not np.isfinite(best_constraint_violation) else float(best_constraint_violation)
            ),
            "best_constraint_breakdown": best_constraint_breakdown,
        },
    )
