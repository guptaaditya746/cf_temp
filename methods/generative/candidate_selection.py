# part4_candidate_selection.py
# Generative Infilling Counterfactuals — Part 4: Candidate Selection
#
# Responsibilities:
# - enforce anomaly validity: reconstruction_score(x_cf) ≤ τ
# - combine constraint results (Part 3) with score validity
# - support single-objective and multi-objective (Pareto / lexicographic) selection
# - return the best counterfactual (or None)
#
# This is the ONLY place where the anomaly detector is queried.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------
# Interfaces
# ----------------------------
AnomalyScoreFn = Callable[[np.ndarray], float]  # x_cf (L,F) -> score


# ----------------------------
# Config
# ----------------------------
@dataclass
class SelectionConfig:
    threshold: float

    # mode
    mode: str = "single"  # "single" | "pareto" | "lexicographic"

    # single-objective weights
    w_score_excess: float = 10.0
    w_mask_size: float = 1.0
    w_soft_penalty: float = 1.0

    # lexicographic priority (earlier = higher priority)
    lexicographic_order: Tuple[str, ...] = (
        "score_excess",
        "mask_size",
        "soft_penalty",
    )

    # safety
    allow_equal_threshold: bool = True


# ----------------------------
# Output
# ----------------------------
@dataclass
class SelectionResult:
    x_cf: np.ndarray
    score: float
    meta: Dict[str, Any]


# ----------------------------
# Selector
# ----------------------------
class CandidateSelector:
    def __init__(self, cfg: SelectionConfig, score_fn: AnomalyScoreFn):
        self.cfg = cfg
        self.score_fn = score_fn

    def select(
        self,
        candidates: List[Any],  # InfillCandidate-like
        constraint_results: List[Any],  # ConstraintResult
    ) -> Optional[SelectionResult]:
        if not candidates:
            return None
        if len(candidates) != len(constraint_results):
            raise ValueError("Candidates and constraint_results length mismatch")

        # 1) Evaluate anomaly score + filter by feasibility
        records = []
        for cand, cres in zip(candidates, constraint_results):
            if not cres.feasible:
                continue

            x_cf_np = (
                cand.x_cf.detach().cpu().numpy()
                if hasattr(cand.x_cf, "detach")
                else np.asarray(cand.x_cf)
            )
            score = float(self.score_fn(x_cf_np))

            if self.cfg.allow_equal_threshold:
                valid = score <= self.cfg.threshold
            else:
                valid = score < self.cfg.threshold

            if not valid:
                continue

            rec = {
                "candidate": cand,
                "constraint": cres,
                "score": score,
                "score_excess": max(0.0, score - self.cfg.threshold),
                "mask_size": int(cand.mask_size),
                "soft_penalty": float(cres.total_soft_penalty),
            }
            records.append(rec)

        if not records:
            return None

        # 2) Selection strategy
        mode = self.cfg.mode.lower()
        if mode == "single":
            best = self._single_objective(records)
        elif mode == "pareto":
            best = self._pareto_select(records)
        elif mode == "lexicographic":
            best = self._lexicographic_select(records)
        else:
            raise ValueError(f"Unknown selection mode: {self.cfg.mode}")

        cand = best["candidate"]
        meta = {
            "masked_segment": cand.masked_segment,
            "mask_size": cand.mask_size,
            "boundary_smoothness": best["constraint"].soft_metrics.get(
                "curvature", None
            ),
            "constraint_metrics": best["constraint"].soft_metrics,
            "libraries_used": cand.meta.get("libraries_used", []),
            "sampling_seed": cand.seed,
            "method": "generative_infilling",
            "selection_mode": self.cfg.mode,
        }

        return SelectionResult(
            x_cf=cand.x_cf.detach().cpu().numpy()
            if hasattr(cand.x_cf, "detach")
            else np.asarray(cand.x_cf),
            score=float(best["score"]),
            meta=meta,
        )

    # ----------------------------
    # Strategies
    # ----------------------------
    def _single_objective(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = self.cfg
        best = None
        best_loss = float("inf")

        for r in records:
            loss = (
                cfg.w_score_excess * r["score_excess"]
                + cfg.w_mask_size * r["mask_size"]
                + cfg.w_soft_penalty * r["soft_penalty"]
            )
            if loss < best_loss:
                best_loss = loss
                best = r

        return best

    def _pareto_select(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple Pareto filtering:
        Objectives: minimize (score_excess, mask_size, soft_penalty)
        Tie-break: minimal mask_size, then minimal soft_penalty
        """
        objs = np.array(
            [[r["score_excess"], r["mask_size"], r["soft_penalty"]] for r in records]
        )

        is_dominated = np.zeros(len(records), dtype=bool)
        for i in range(len(records)):
            for j in range(len(records)):
                if i == j:
                    continue
                if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    is_dominated[i] = True
                    break

        pareto = [r for r, d in zip(records, is_dominated) if not d]
        pareto.sort(key=lambda r: (r["mask_size"], r["soft_penalty"]))
        return pareto[0]

    def _lexicographic_select(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        order = self.cfg.lexicographic_order

        def key_fn(r: Dict[str, Any]):
            return tuple(r[o] for o in order)

        records.sort(key=key_fn)
        return records[0]


# ----------------------------
# Minimal self-test
# ----------------------------
if __name__ == "__main__":
    # Dummy score: lower is better
    def score_fn(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x)))

    # Fake candidates / constraints
    class _Cand:
        def __init__(self, v, mask_size, seed):
            self.x_cf = np.ones((10, 2)) * v
            self.mask_size = mask_size
            self.masked_segment = (2, 5)
            self.seed = seed
            self.meta = {"libraries_used": ["torch", "numpy"]}

    class _CR:
        def __init__(self, feasible, pen):
            self.feasible = feasible
            self.total_soft_penalty = pen
            self.soft_metrics = {"curvature": pen}

    cands = [_Cand(0.1, 10, 1), _Cand(0.2, 5, 2), _Cand(0.05, 20, 3)]
    cres = [_CR(True, 1.0), _CR(True, 2.0), _CR(True, 0.5)]

    cfg = SelectionConfig(threshold=0.15, mode="single")
    sel = CandidateSelector(cfg, score_fn)
    res = sel.select(cands, cres)

    print("Selected score:", res.score)
    print("Mask size:", res.meta["mask_size"])
