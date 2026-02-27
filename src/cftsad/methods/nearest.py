from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.candidates import compute_candidate_metrics, rank_candidates
from cftsad.core.constraints import apply_constraints
from cftsad.core.constraints_v2 import apply_constraints_v2
from cftsad.core.postprocess import build_explainability_meta
from cftsad.core.distances import window_mse_distance
from cftsad.core.scoring import reconstruction_score
from cftsad.types import CFFailure, CFResult


def _window_weighted_distance(
    x: np.ndarray,
    y: np.ndarray,
    feature_scale: np.ndarray,
    timestep_weights: np.ndarray | None = None,
) -> float:
    diff = (x - y) / feature_scale[np.newaxis, :]
    sq = diff * diff
    if timestep_weights is not None:
        sq = sq * timestep_weights[:, np.newaxis]
    return float(np.mean(sq))


def generate_nearest(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray,
    threshold: float,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
    top_k: int = 10,
    alpha_steps: int = 11,
    donor_filter_factor: float = 1.0,
    use_weighted_distance: bool = True,
    use_constraints_v2: bool = False,
    max_delta_per_step: float | None = None,
    relational_linear: Dict[str, tuple[int, int, float]] | None = None,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    score_before = reconstruction_score(model, x)
    thr = float(threshold)

    donor_scores = np.asarray(
        [reconstruction_score(model, donor) for donor in normal_core],
        dtype=np.float64,
    )
    strict_thr = float(thr) * float(donor_filter_factor)
    donor_mask = donor_scores <= strict_thr
    donor_pool_idx = np.where(donor_mask)[0]
    if donor_pool_idx.size == 0:
        donor_pool_idx = np.arange(normal_core.shape[0], dtype=np.int64)

    if use_weighted_distance:
        flat = normal_core.reshape(-1, normal_core.shape[-1])
        scale = np.std(flat, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        dists = np.asarray(
            [
                _window_weighted_distance(x, normal_core[i], feature_scale=scale)
                for i in donor_pool_idx
            ],
            dtype=np.float64,
        )
    else:
        dists = np.asarray(
            [window_mse_distance(x, normal_core[i]) for i in donor_pool_idx],
            dtype=np.float64,
        )

    k = max(1, min(int(top_k), int(donor_pool_idx.size)))
    shortlist_local = np.argpartition(dists, k - 1)[:k]
    shortlist_local = shortlist_local[np.argsort(dists[shortlist_local])]
    shortlist_idx = donor_pool_idx[shortlist_local]

    a_steps = max(2, int(alpha_steps))
    alphas = np.linspace(0.0, 1.0, a_steps)
    candidates = []
    attempts = 0
    best_constraint_violation = np.inf
    best_constraint_breakdown = None

    for donor_idx in shortlist_idx:
        donor = normal_core[int(donor_idx)]
        donor_distance = float(
            window_mse_distance(x, donor)
            if not use_weighted_distance
            else _window_weighted_distance(x, donor, feature_scale=scale)
        )
        for alpha in alphas:
            attempts += 1
            mixed = (1.0 - float(alpha)) * x + float(alpha) * donor
            if use_constraints_v2:
                x_cf, c_v, c_break = apply_constraints_v2(
                    mixed,
                    x,
                    immutable_features=immutable_features,
                    bounds=bounds,
                    max_delta_per_step=max_delta_per_step,
                    relational_linear=relational_linear,
                )
            else:
                x_cf = apply_constraints(mixed, x, immutable_features, bounds)
                c_v, c_break = 0.0, None
            if c_v < best_constraint_violation:
                best_constraint_violation = float(c_v)
                best_constraint_breakdown = c_break
            score_after = reconstruction_score(model, x_cf)
            cand = compute_candidate_metrics(
                x=x,
                x_cf=x_cf,
                score_cf=score_after,
                normal_core=normal_core,
            )
            cand.diagnostics = {
                "donor_idx": int(donor_idx),
                "alpha": float(alpha),
                "donor_distance": donor_distance,
                "constraint_violation": float(c_v),
                "constraint_breakdown": c_break,
            }
            candidates.append(cand)

    ranked = rank_candidates(candidates, threshold=thr)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    if ranked and ranked[0].score_cf <= thr:
        best = ranked[0]
        meta = {
            "donor_idx": int(best.diagnostics["donor_idx"]),
            "alpha": float(best.diagnostics["alpha"]),
            "donor_distance": float(best.diagnostics["donor_distance"]),
            "score_before": score_before,
            "score_after": float(best.score_cf),
            "strict_donor_threshold": strict_thr,
            "shortlist_size": int(k),
            "n_attempts": int(attempts),
            "constraint_violation": float(best.diagnostics["constraint_violation"]),
            "constraint_breakdown": best.diagnostics["constraint_breakdown"],
            "runtime_ms": runtime_ms,
        }
        meta.update(build_explainability_meta(x, best.x_cf))
        return CFResult(x_cf=best.x_cf, score_cf=float(best.score_cf), meta=meta)

    return CFFailure(
        reason="no_valid_cf",
        message="Nearest donor shortlist did not satisfy threshold.",
        diagnostics={
            "score_before": score_before,
            "best_score_after": float(ranked[0].score_cf) if ranked else None,
            "threshold": thr,
            "strict_donor_threshold": strict_thr,
            "shortlist_size": int(k),
            "n_attempts": int(attempts),
            "best_constraint_violation": (
                None if not np.isfinite(best_constraint_violation) else float(best_constraint_violation)
            ),
            "best_constraint_breakdown": best_constraint_breakdown,
            "runtime_ms": runtime_ms,
        },
    )
