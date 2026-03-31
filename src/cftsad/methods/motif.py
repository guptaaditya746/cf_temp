from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.candidates import compute_candidate_metrics, rank_candidates
from cftsad.core.constraints import apply_constraints
from cftsad.core.constraints_v2 import apply_constraints_v2
from cftsad.core.localization import (
    build_segment_groups as _build_segment_groups,
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


def _z_normalize(segment: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(segment, axis=0, keepdims=True)
    std = np.std(segment, axis=0, keepdims=True)
    return (segment - mean) / (std + eps)


def _build_motif_index(
    normal_core: np.ndarray,
    motif_length: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    vectors = []
    sources: list[tuple[int, int]] = []
    K, L, _ = normal_core.shape

    if motif_length > L:
        return np.empty((0, motif_length), dtype=np.float64), sources

    for donor_idx in range(K):
        donor = normal_core[donor_idx]
        for start in range(0, L - motif_length + 1):
            seg = donor[start : start + motif_length]
            zn = _z_normalize(seg).reshape(-1)
            vectors.append(zn)
            sources.append((donor_idx, start))

    if not vectors:
        return np.empty((0, motif_length), dtype=np.float64), sources
    return np.asarray(vectors, dtype=np.float64), sources


def _fit_affine(target: np.ndarray, motif: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    x = motif.reshape(-1).astype(np.float64)
    y = target.reshape(-1).astype(np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var_x = float(np.mean((x - x_mean) ** 2))
    if var_x < eps:
        return 1.0, float(y_mean - x_mean)
    cov = float(np.mean((x - x_mean) * (y - y_mean)))
    a = cov / (var_x + eps)
    b = y_mean - a * x_mean
    return float(a), float(b)


def _boundary_penalty(
    x: np.ndarray,
    candidate_seg: np.ndarray,
    start: int,
    end: int,
) -> float:
    penalty = 0.0
    if start > 0:
        penalty += float(np.mean((x[start - 1] - candidate_seg[0]) ** 2))
    if end < x.shape[0] - 1:
        penalty += float(np.mean((candidate_seg[-1] - x[end + 1]) ** 2))
    return penalty


def generate_motif(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray,
    threshold: float,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
    top_k: int = 5,
    n_segments: int = 4,
    length_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    context_weight: float = 0.2,
    use_affine_fit: bool = True,
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

    base_segments, localization_errors, raw_errors, reference = _detect_candidate_segments(
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
    if not base_segments:
        return CFFailure(
            reason="segment_detection_failed",
            message="Unable to detect anomalous segment for motif substitution.",
            diagnostics={"score_before": score_before, "threshold": thr},
        )

    segment_candidates: list[tuple[int, int]] = []
    for start, end in base_segments:
        base_len = end - start + 1
        center = (start + end) // 2
        for fac in length_factors:
            mlen = max(3, int(round(base_len * float(fac))))
            half = mlen // 2
            s = max(0, center - half)
            e = min(x.shape[0] - 1, s + mlen - 1)
            s = max(0, e - mlen + 1)
            segment_candidates.append((int(s), int(e)))
    segment_candidates = sorted(
        set(segment_candidates),
        key=lambda se: (
            -float(np.mean(localization_errors[se[0] : se[1] + 1])),
            -int(se[1] - se[0] + 1),
            int(se[0]),
        ),
    )
    segment_scores = {
        tuple(seg): float(np.mean(localization_errors[seg[0] : seg[1] + 1]))
        for seg in segment_candidates
    }
    segment_groups = _build_segment_groups(
        segment_candidates,
        segment_scores,
        allow_pair_search=allow_pair_search,
        max_pair_groups=max_pair_groups,
    )

    all_candidates = []
    attempted = 0
    motif_lengths = set()
    best_constraint_violation = np.inf
    best_constraint_breakdown = None
    motif_search_summary = []
    pair_match_cap = min(int(top_k), 4)

    for group in segment_groups:
        group_summary = {
            "group_type": "pair" if len(group) == 2 else "single",
            "segment_count": int(len(group)),
            "group_score": float(sum(segment_scores.get(tuple(seg), 0.0) for seg in group)),
            "segments": [],
        }
        match_groups = []

        for start, end in group:
            motif_length = end - start + 1
            motif_lengths.add(int(motif_length))
            motif_vectors, sources = _build_motif_index(normal_core, motif_length)
            if motif_vectors.shape[0] == 0:
                match_groups = []
                break

            query = _z_normalize(x[start : end + 1]).reshape(-1).astype(np.float64)
            shape_dist = np.linalg.norm(motif_vectors - query[None, :], axis=1)
            k_full = max(1, min(int(top_k), shape_dist.shape[0]))
            top_idx = np.argpartition(shape_dist, k_full - 1)[:k_full]
            top_idx = top_idx[np.argsort(shape_dist[top_idx])]
            k_effective = k_full if len(group) == 1 else max(1, min(pair_match_cap, k_full))
            top_idx = top_idx[:k_effective]
            top_matches = [
                {
                    "donor_idx": int(sources[int(idx)][0]),
                    "donor_start": int(sources[int(idx)][1]),
                    "shape_distance": float(shape_dist[int(idx)]),
                }
                for idx in top_idx[: min(3, len(top_idx))].tolist()
            ]
            group_summary["segments"].append(
                {
                    "segment_start": int(start),
                    "segment_end": int(end),
                    "motif_length": int(motif_length),
                    "touches_left_boundary": bool(start == 0),
                    "touches_right_boundary": bool(end == x.shape[0] - 1),
                    "segment_mean_localizer_error": float(np.mean(localization_errors[start : end + 1])),
                    "segment_max_localizer_error": float(np.max(localization_errors[start : end + 1])),
                    "segment_mean_raw_error": float(np.mean(raw_errors[start : end + 1])),
                    "segment_max_raw_error": float(np.max(raw_errors[start : end + 1])),
                    "n_available_motifs": int(shape_dist.shape[0]),
                    "top_matches": top_matches,
                }
            )
            match_groups.append(
                [
                    {
                        "segment": (int(start), int(end)),
                        "motif_length": int(motif_length),
                        "donor_idx": int(sources[int(idx)][0]),
                        "donor_start": int(sources[int(idx)][1]),
                        "shape_distance": float(shape_dist[int(idx)]),
                    }
                    for idx in top_idx.tolist()
                ]
            )

        motif_search_summary.append(group_summary)
        if not match_groups:
            continue

        for combo in itertools.product(*match_groups):
            attempted += 1
            candidate = np.array(x, copy=True)
            boundary_total = 0.0
            shape_total = 0.0
            affine_params = []
            motif_sources = []

            for choice in sorted(combo, key=lambda item: item["segment"][0]):
                start, end = choice["segment"]
                donor_start = int(choice["donor_start"])
                donor_end = donor_start + int(choice["motif_length"])
                donor_seg = np.array(
                    normal_core[int(choice["donor_idx"]), donor_start:donor_end],
                    copy=True,
                )
                if use_affine_fit:
                    a, b = _fit_affine(x[start : end + 1], donor_seg)
                    donor_seg = a * donor_seg + b
                else:
                    a, b = 1.0, 0.0

                boundary = _boundary_penalty(x, donor_seg, start, end)
                candidate[start : end + 1] = donor_seg
                boundary_total += float(boundary)
                shape_total += float(choice["shape_distance"])
                affine_params.append({"a": float(a), "b": float(b)})
                motif_sources.append(
                    {
                        "donor_idx": int(choice["donor_idx"]),
                        "segment_start": int(donor_start),
                        "segment_end": int(donor_end - 1),
                    }
                )

            rank_score = float(shape_total + float(context_weight) * boundary_total)
            if use_constraints_v2:
                candidate, c_v, c_break = apply_constraints_v2(
                    candidate,
                    x,
                    immutable_features=immutable_features,
                    bounds=bounds,
                    max_delta_per_step=max_delta_per_step,
                    relational_linear=relational_linear,
                )
            else:
                candidate = apply_constraints(candidate, x, immutable_features, bounds)
                c_v, c_break = 0.0, None
            if c_v < best_constraint_violation:
                best_constraint_violation = float(c_v)
                best_constraint_breakdown = c_break

            score_after = _score_candidate(model, candidate, score_fn)
            cand = compute_candidate_metrics(
                x=x,
                x_cf=candidate,
                score_cf=score_after,
                normal_core=normal_core,
            )
            cand.diagnostics = {
                "segment_count": int(len(combo)),
                "segment_group": [list(choice["segment"]) for choice in combo],
                "motif_lengths": [int(choice["motif_length"]) for choice in combo],
                "chosen_motif_sources": motif_sources,
                "shape_distance_sum": float(shape_total),
                "boundary_penalty_sum": float(boundary_total),
                "motif_rank_score": float(rank_score),
                "affine_params": affine_params,
                "constraint_violation": float(c_v),
                "constraint_breakdown": c_break,
            }
            all_candidates.append(cand)

    ranked = rank_candidates(all_candidates, threshold=thr)
    best = ranked[0] if ranked else None
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    common_meta = {
        "segment_candidates": motif_search_summary,
        "motif_lengths_considered": sorted(int(v) for v in motif_lengths),
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
        "n_segments_considered": int(len(segment_candidates)),
        "n_segment_groups_considered": int(len(segment_groups)),
        "n_candidates_evaluated": int(attempted),
        "best_candidate_summary": build_candidate_summary(x, best, threshold=thr),
        "runtime_ms": runtime_ms,
        "score_source": "custom" if score_fn is not None else "reconstruction_score",
    }
    if best is not None and best.score_cf <= thr:
        first_seg = best.diagnostics["segment_group"][0]
        meta = {
            "motif_length": int(best.diagnostics["motif_lengths"][0]),
            "segment_start": int(first_seg[0]),
            "segment_end": int(first_seg[1]),
            "segment_count": int(best.diagnostics["segment_count"]),
            "segment_group": best.diagnostics["segment_group"],
            "chosen_motif_sources": best.diagnostics["chosen_motif_sources"],
            "shape_distance_sum": float(best.diagnostics["shape_distance_sum"]),
            "boundary_penalty_sum": float(best.diagnostics["boundary_penalty_sum"]),
            "motif_rank_score": float(best.diagnostics["motif_rank_score"]),
            "affine_params": best.diagnostics["affine_params"],
            "constraint_violation": float(best.diagnostics["constraint_violation"]),
            "constraint_breakdown": best.diagnostics["constraint_breakdown"],
        }
        meta.update(build_score_summary(score_before, float(best.score_cf), thr))
        meta.update(common_meta)
        meta.update(build_explainability_meta(x, best.x_cf))
        return CFResult(x_cf=best.x_cf, score_cf=float(best.score_cf), meta=meta)

    return CFFailure(
        reason="no_valid_cf",
        message="Motif search did not produce a valid counterfactual.",
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
