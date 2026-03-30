from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.candidates import compute_candidate_metrics, rank_candidates
from cftsad.core.constraints import apply_constraints
from cftsad.core.constraints_v2 import apply_constraints_v2
from cftsad.core.distances import window_mse_distance
from cftsad.core.localization import (
    build_segment_groups as _build_segment_groups,
    detect_anomalous_segment as _detect_anomalous_segment,
    detect_candidate_segments as _detect_candidate_segments,
)
from cftsad.core.postprocess import (
    build_candidate_summary,
    build_explainability_meta,
    build_score_summary,
)
from cftsad.core.scoring import reconstruction_score
from cftsad.types import CFFailure, CFResult


def _score_candidate(
    model: object,
    x_cf: np.ndarray,
    score_fn: Optional[Callable[[np.ndarray], float]],
) -> float:
    if score_fn is not None:
        return float(score_fn(np.asarray(x_cf, dtype=np.float64)))
    return reconstruction_score(model, x_cf)


def detect_anomalous_segment(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray | None = None,
    *,
    top_fraction: float = 0.1,
    padding: int = 2,
    min_length: int | None = None,
    normalize_errors: bool = True,
) -> tuple[int, int] | None:
    return _detect_anomalous_segment(
        model,
        x,
        normal_core=normal_core,
        top_fraction=top_fraction,
        padding=padding,
        min_length=min_length,
        normalize_errors=normalize_errors,
    )


def detect_candidate_segments(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray | None = None,
    *,
    n_segments: int = 4,
    length_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    top_fraction: float = 0.1,
    padding: int = 2,
    min_length: int | None = None,
    normalize_errors: bool = True,
) -> list[tuple[int, int]]:
    segments, _, _, _ = _detect_candidate_segments(
        model,
        x,
        normal_core=normal_core,
        n_segments=n_segments,
        length_factors=length_factors,
        top_fraction=top_fraction,
        padding=padding,
        min_length=min_length,
        normalize_errors=normalize_errors,
    )
    return segments


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
    length_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    top_k_donors: int = 8,
    context_width: int = 2,
    crossfade_width: int = 3,
    allow_pair_search: bool = True,
    max_pair_groups: int = 4,
    localizer_top_fraction: float = 0.1,
    localizer_padding: int = 2,
    normalize_errors: bool = True,
    use_constraints_v2: bool = False,
    max_delta_per_step: float | None = None,
    relational_linear: Dict[str, tuple[int, int, float]] | None = None,
    score_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    score_before = _score_candidate(model, x, score_fn)
    thr = float(threshold)
    segments, localization_errors, raw_errors, reference = _detect_candidate_segments(
        model,
        x,
        normal_core=normal_core,
        n_segments=n_segments,
        length_factors=length_factors,
        top_fraction=localizer_top_fraction,
        padding=localizer_padding,
        min_length=max(5, x.shape[0] // 20),
        normalize_errors=normalize_errors,
    )
    if not segments:
        return CFFailure(
            reason="segment_detection_failed",
            message="Unable to detect anomalous segment.",
            diagnostics={"threshold": thr, "score_before": score_before},
        )

    segment_scores = {
        tuple(seg): float(np.mean(localization_errors[seg[0] : seg[1] + 1]))
        for seg in segments
    }
    segment_groups = _build_segment_groups(
        segments,
        segment_scores,
        allow_pair_search=allow_pair_search,
        max_pair_groups=max_pair_groups,
    )

    candidates = []
    attempted = 0
    best_constraint_violation = np.inf
    best_constraint_breakdown = None
    segment_search_summary = []
    pair_donor_cap = min(int(top_k_donors), 4)

    for group in segment_groups:
        donor_choices = []
        group_summary = {
            "group_type": "pair" if len(group) == 2 else "single",
            "segment_count": int(len(group)),
            "group_score": float(sum(segment_scores.get(tuple(seg), 0.0) for seg in group)),
            "segments": [],
        }

        for start, end in group:
            c0 = max(0, start - int(context_width))
            c1 = min(x.shape[0] - 1, end + int(context_width))
            seg_x = x[c0 : c1 + 1]
            donor_dists = np.asarray(
                [window_mse_distance(seg_x, donor[c0 : c1 + 1]) for donor in normal_core],
                dtype=np.float64,
            )
            k_full = max(1, min(int(top_k_donors), int(donor_dists.shape[0])))
            top_idx = np.argpartition(donor_dists, k_full - 1)[:k_full]
            top_idx = top_idx[np.argsort(donor_dists[top_idx])]
            k_effective = k_full if len(group) == 1 else max(1, min(pair_donor_cap, k_full))
            top_idx = top_idx[:k_effective]

            donor_choices.append(
                [
                    {
                        "segment": (int(start), int(end)),
                        "context_start": int(c0),
                        "context_end": int(c1),
                        "donor_idx": int(donor_idx),
                        "distance": float(donor_dists[int(donor_idx)]),
                    }
                    for donor_idx in top_idx.tolist()
                ]
            )
            group_summary["segments"].append(
                {
                    "segment_start": int(start),
                    "segment_end": int(end),
                    "segment_length": int(end - start + 1),
                    "touches_left_boundary": bool(start == 0),
                    "touches_right_boundary": bool(end == x.shape[0] - 1),
                    "context_start": int(c0),
                    "context_end": int(c1),
                    "segment_mean_localizer_error": float(np.mean(localization_errors[start : end + 1])),
                    "segment_max_localizer_error": float(np.max(localization_errors[start : end + 1])),
                    "segment_mean_raw_error": float(np.mean(raw_errors[start : end + 1])),
                    "segment_max_raw_error": float(np.max(raw_errors[start : end + 1])),
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
        segment_search_summary.append(group_summary)

        for donor_combo in itertools.product(*donor_choices):
            attempted += 1
            x_cf = np.array(x, copy=True)
            for choice in sorted(donor_combo, key=lambda item: item["segment"][0]):
                start, end = choice["segment"]
                x_cf[start : end + 1] = normal_core[int(choice["donor_idx"]), start : end + 1]

            if smoothing:
                for choice in sorted(donor_combo, key=lambda item: item["segment"][0]):
                    start, end = choice["segment"]
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
                "segment_count": int(len(donor_combo)),
                "segment_group": [list(choice["segment"]) for choice in donor_combo],
                "context_group": [
                    [int(choice["context_start"]), int(choice["context_end"])]
                    for choice in donor_combo
                ],
                "donor_indices": [int(choice["donor_idx"]) for choice in donor_combo],
                "segment_distances": [float(choice["distance"]) for choice in donor_combo],
                "constraint_violation": float(c_v),
                "constraint_breakdown": c_break,
            }
            candidates.append(cand)

    ranked = rank_candidates(candidates, threshold=thr)
    best = ranked[0] if ranked else None
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    common_meta = {
        "segment_candidates": segment_search_summary,
        "localizer_peak_timestep": int(np.argmax(localization_errors)),
        "localizer_peak_error": float(np.max(localization_errors)),
        "raw_peak_timestep": int(np.argmax(raw_errors)),
        "raw_peak_error": float(np.max(raw_errors)),
        "normalize_errors_used": bool(normalize_errors),
        "localizer_top_fraction": float(localizer_top_fraction),
        "localizer_padding": int(localizer_padding),
        "calibration_error_reference": (
            None
            if reference is None
            else {
                "mean_min": float(np.min(reference["mean"])),
                "mean_median": float(np.median(reference["mean"])),
                "mean_max": float(np.max(reference["mean"])),
                "std_min": float(np.min(reference["std"])),
                "std_median": float(np.median(reference["std"])),
                "std_max": float(np.max(reference["std"])),
            }
        ),
        "smoothing_used": bool(smoothing),
        "n_segments_considered": int(len(segments)),
        "n_segment_groups_considered": int(len(segment_groups)),
        "n_candidates_evaluated": int(attempted),
        "best_candidate_summary": build_candidate_summary(x, best, threshold=thr),
        "runtime_ms": runtime_ms,
        "score_source": "custom" if score_fn is not None else "reconstruction_score",
    }
    if best is not None and best.score_cf <= thr:
        first_seg = best.diagnostics["segment_group"][0]
        meta = {
            "segment_start": int(first_seg[0]),
            "segment_end": int(first_seg[1]),
            "segment_count": int(best.diagnostics["segment_count"]),
            "segment_group": best.diagnostics["segment_group"],
            "context_group": best.diagnostics["context_group"],
            "donor_indices": best.diagnostics["donor_indices"],
            "segment_distances": best.diagnostics["segment_distances"],
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
