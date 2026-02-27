from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class Candidate:
    x_cf: np.ndarray
    score_cf: float
    proximity: float
    sparsity: float
    plausibility: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def deduplicate_candidates(
    candidates: Iterable[Candidate],
    decimals: int = 8,
) -> List[Candidate]:
    seen: set[bytes] = set()
    out: List[Candidate] = []
    for cand in candidates:
        key = np.round(np.asarray(cand.x_cf, dtype=np.float64), decimals=decimals).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def prune_candidates(
    candidates: Sequence[Candidate],
    max_candidates: int | None,
    threshold: float,
) -> List[Candidate]:
    ranked = rank_candidates(candidates, threshold=threshold)
    if max_candidates is None:
        return ranked
    k = max(1, int(max_candidates))
    return ranked[:k]


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


def evaluate_candidate_pool(
    x: np.ndarray,
    candidates_x: Sequence[np.ndarray],
    score_fn: Callable[[np.ndarray], float],
    threshold: float,
    normal_core: np.ndarray | None = None,
    max_candidates: int | None = None,
    dedup_decimals: int = 8,
    batch_size: int | None = None,
) -> List[Candidate]:
    """
    Build+rank candidates from raw windows.
    Batching is API-level only (loop chunking) and does not enforce vectorized score_fn.
    """
    windows = [np.asarray(v, dtype=np.float64) for v in candidates_x]
    if batch_size is None or batch_size < 1:
        batch_size = len(windows) if windows else 1

    all_cands: List[Candidate] = []
    for i in range(0, len(windows), int(batch_size)):
        chunk = windows[i : i + int(batch_size)]
        for x_cf in chunk:
            score = float(score_fn(x_cf))
            all_cands.append(
                compute_candidate_metrics(
                    x=x,
                    x_cf=x_cf,
                    score_cf=score,
                    normal_core=normal_core,
                )
            )

    all_cands = deduplicate_candidates(all_cands, decimals=dedup_decimals)
    return prune_candidates(all_cands, max_candidates=max_candidates, threshold=threshold)
