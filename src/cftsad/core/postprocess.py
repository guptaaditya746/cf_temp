from __future__ import annotations

from typing import Any, Dict

import numpy as np


def build_explainability_meta(
    x: np.ndarray,
    x_cf: np.ndarray,
    top_k: int = 3,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    x_cf = np.asarray(x_cf, dtype=np.float64)
    delta = x_cf - x
    abs_delta = np.abs(delta)

    feature_change = np.sum(abs_delta, axis=0)
    timestep_change = np.sum(abs_delta, axis=1)
    edit_mask = abs_delta > float(eps)

    feature_order = np.argsort(-feature_change)
    timestep_order = np.argsort(-timestep_change)
    k = max(1, int(top_k))

    return {
        "changed_features": [int(i) for i in feature_order.tolist()],
        "changed_timesteps": [int(i) for i in timestep_order.tolist()],
        "top_changed_features": [
            {"feature": int(i), "abs_delta_sum": float(feature_change[i])}
            for i in feature_order[:k]
        ],
        "top_changed_timesteps": [
            {"timestep": int(i), "abs_delta_sum": float(timestep_change[i])}
            for i in timestep_order[:k]
        ],
        "edit_mask": edit_mask,
        "delta_l1": float(np.sum(abs_delta)),
        "delta_l2": float(np.sqrt(np.sum(delta * delta))),
    }


def build_score_summary(
    score_before: float,
    score_after: float | None,
    threshold: float,
) -> Dict[str, Any]:
    score_before = float(score_before)
    threshold = float(threshold)
    out: Dict[str, Any] = {
        "score_before": score_before,
        "threshold": threshold,
        "threshold_gap_before": float(score_before - threshold),
    }

    if score_after is None:
        out.update(
            {
                "score_after": None,
                "score_delta": None,
                "relative_score_delta": None,
                "threshold_gap_after": None,
            }
        )
        return out

    score_after = float(score_after)
    score_delta = float(score_before - score_after)
    out.update(
        {
            "score_after": score_after,
            "score_delta": score_delta,
            "relative_score_delta": (
                None if abs(score_before) < 1e-12 else float(score_delta / abs(score_before))
            ),
            "threshold_gap_after": float(score_after - threshold),
        }
    )
    return out


def build_candidate_summary(
    x: np.ndarray,
    candidate: Any,
    threshold: float,
    top_k: int = 3,
) -> Dict[str, Any] | None:
    if candidate is None:
        return None

    out = {
        "score_cf": float(candidate.score_cf),
        "proximity": float(candidate.proximity),
        "sparsity": float(candidate.sparsity),
        "plausibility": float(candidate.plausibility),
        "threshold_gap_after": float(float(candidate.score_cf) - float(threshold)),
    }
    out.update(build_explainability_meta(x, candidate.x_cf, top_k=top_k))
    if getattr(candidate, "diagnostics", None):
        out["search_diagnostics"] = candidate.diagnostics
    return out
