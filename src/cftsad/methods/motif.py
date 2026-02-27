from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.candidates import compute_candidate_metrics, rank_candidates
from cftsad.core.constraints import apply_constraints
from cftsad.core.constraints_v2 import apply_constraints_v2
from cftsad.core.postprocess import build_explainability_meta
from cftsad.core.scoring import reconstruction_score
from cftsad.methods.segment import detect_candidate_segments
from cftsad.types import CFFailure, CFResult


def _z_normalize(segment: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(segment, axis=0, keepdims=True)
    std = np.std(segment, axis=0, keepdims=True)
    return (segment - mean) / (std + eps)


def _build_motif_index(normal_core: np.ndarray, motif_length: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
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
    length_factors: tuple[float, ...] = (0.75, 1.0, 1.25),
    context_weight: float = 0.2,
    use_affine_fit: bool = True,
    use_constraints_v2: bool = False,
    max_delta_per_step: float | None = None,
    relational_linear: Dict[str, tuple[int, int, float]] | None = None,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    score_before = reconstruction_score(model, x)
    thr = float(threshold)

    base_segments = detect_candidate_segments(model, x, n_segments=n_segments)
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
    segment_candidates = sorted(set(segment_candidates))

    all_candidates = []
    attempted = 0
    motif_lengths = set()
    best_constraint_violation = np.inf
    best_constraint_breakdown = None

    for start, end in segment_candidates:
        motif_length = end - start + 1
        motif_lengths.add(int(motif_length))
        motif_vectors, sources = _build_motif_index(normal_core, motif_length)
        if motif_vectors.shape[0] == 0:
            continue

        query = _z_normalize(x[start : end + 1]).reshape(-1).astype(np.float64)
        shape_dist = np.linalg.norm(motif_vectors - query[None, :], axis=1)
        k = max(1, min(int(top_k), shape_dist.shape[0]))
        top_idx = np.argpartition(shape_dist, k - 1)[:k]
        top_idx = top_idx[np.argsort(shape_dist[top_idx])]

        for idx in top_idx:
            attempted += 1
            donor_idx, donor_start = sources[int(idx)]
            donor_end = donor_start + motif_length

            donor_seg = np.array(
                normal_core[int(donor_idx), donor_start:donor_end],
                copy=True,
            )
            if use_affine_fit:
                a, b = _fit_affine(x[start : end + 1], donor_seg)
                donor_seg = a * donor_seg + b
            else:
                a, b = 1.0, 0.0

            boundary = _boundary_penalty(x, donor_seg, start, end)
            rank_score = float(shape_dist[int(idx)]) + float(context_weight) * boundary

            candidate = np.array(x, copy=True)
            candidate[start : end + 1] = donor_seg
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
            score_after = reconstruction_score(model, candidate)

            cand = compute_candidate_metrics(
                x=x,
                x_cf=candidate,
                score_cf=score_after,
                normal_core=normal_core,
            )
            cand.diagnostics = {
                "motif_length": int(motif_length),
                "segment_start": int(start),
                "segment_end": int(end),
                "donor_idx": int(donor_idx),
                "donor_segment_start": int(donor_start),
                "donor_segment_end": int(donor_end - 1),
                "shape_distance": float(shape_dist[int(idx)]),
                "boundary_penalty": float(boundary),
                "motif_rank_score": float(rank_score),
                "affine_a": float(a),
                "affine_b": float(b),
                "constraint_violation": float(c_v),
                "constraint_breakdown": c_break,
            }
            all_candidates.append(cand)

    ranked = rank_candidates(all_candidates, threshold=thr)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    if ranked and ranked[0].score_cf <= thr:
        best = ranked[0]
        meta = {
            "motif_length": int(best.diagnostics["motif_length"]),
            "segment_start": int(best.diagnostics["segment_start"]),
            "segment_end": int(best.diagnostics["segment_end"]),
            "chosen_motif_source": {
                "donor_idx": int(best.diagnostics["donor_idx"]),
                "segment_start": int(best.diagnostics["donor_segment_start"]),
                "segment_end": int(best.diagnostics["donor_segment_end"]),
            },
            "shape_distance": float(best.diagnostics["shape_distance"]),
            "boundary_penalty": float(best.diagnostics["boundary_penalty"]),
            "motif_rank_score": float(best.diagnostics["motif_rank_score"]),
            "affine_a": float(best.diagnostics["affine_a"]),
            "affine_b": float(best.diagnostics["affine_b"]),
            "score_before": score_before,
            "score_after": float(best.score_cf),
            "n_segments_considered": int(len(segment_candidates)),
            "n_candidates_evaluated": int(attempted),
            "constraint_violation": float(best.diagnostics["constraint_violation"]),
            "constraint_breakdown": best.diagnostics["constraint_breakdown"],
            "runtime_ms": runtime_ms,
        }
        meta.update(build_explainability_meta(x, best.x_cf))
        return CFResult(x_cf=best.x_cf, score_cf=float(best.score_cf), meta=meta)

    return CFFailure(
        reason="no_valid_cf",
        message="Motif search did not produce a valid counterfactual.",
        diagnostics={
            "score_before": score_before,
            "best_score_after": float(ranked[0].score_cf) if ranked else None,
            "threshold": thr,
            "motif_lengths_considered": sorted(int(v) for v in motif_lengths),
            "n_segments_considered": int(len(segment_candidates)),
            "n_candidates_evaluated": int(attempted),
            "best_constraint_violation": (
                None if not np.isfinite(best_constraint_violation) else float(best_constraint_violation)
            ),
            "best_constraint_breakdown": best_constraint_breakdown,
            "runtime_ms": runtime_ms,
        },
    )
