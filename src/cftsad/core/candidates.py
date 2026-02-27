from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np


@dataclass
class Candidate:
    x_cf: np.ndarray
    score_cf: float
    proximity: float
    sparsity: float
    plausibility: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def compute_candidate_metrics(
    x: np.ndarray,
    x_cf: np.ndarray,
    score_cf: float,
    normal_core: np.ndarray | None = None,
) -> Candidate:
    diff = np.asarray(x_cf, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    proximity = float(np.mean(diff * diff))
    changed = np.any(np.abs(diff) > 1e-8, axis=1)
    sparsity = float(np.mean(changed.astype(np.float64)))

    plausibility = 0.0
    if normal_core is not None and normal_core.size > 0:
        d2 = np.mean((normal_core - x_cf[np.newaxis, :, :]) ** 2, axis=(1, 2))
        plausibility = float(np.min(d2))

    return Candidate(
        x_cf=np.asarray(x_cf, dtype=np.float64),
        score_cf=float(score_cf),
        proximity=proximity,
        sparsity=sparsity,
        plausibility=plausibility,
    )


def rank_candidates(
    candidates: Iterable[Candidate],
    threshold: float,
) -> List[Candidate]:
    cands = list(candidates)
    thr = float(threshold)

    def _key(c: Candidate) -> tuple[int, float, float, float, float]:
        is_invalid = 0 if c.score_cf <= thr else 1
        return (
            is_invalid,
            float(c.score_cf),
            float(c.proximity),
            float(c.sparsity),
            float(c.plausibility),
        )

    return sorted(cands, key=_key)
