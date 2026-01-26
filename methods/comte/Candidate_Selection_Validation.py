# Part 5 — Candidate Selection & Validation
# - enforce reconstruction_score(x_cf) <= tau
# - rank feasible CFs by objectives (single-objective or multi-objective)
# - return best counterfactual + full metadata
#
# NOTE:
# - This file assumes you already have outputs from:
#   Part 1: segment candidates
#   Part 2: donor matches
#   Part 3: substitution engine
#   Part 4: constraint evaluator

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

Tensor = torch.Tensor


# -------------------------------
# Config & containers
# -------------------------------


@dataclass
class SelectionConfig:
    mode: str = "single"  # {"single","multi"}

    # validity
    tau: float = 0.0

    # evaluation limits (keep practical)
    max_segments: int = 120
    max_donors_per_segment: int = 5

    # single-objective weights (normalized terms expected)
    alpha_len: float = 1.0
    beta_excess: float = 8.0
    gamma_constraints: float = 2.0
    delta_normalcore_dist: float = 1.0
    eta_boundary: float = 1.0
    zeta_pca: float = 1.0

    # multi-objective selection (lexicographic, practical)
    # order matters: earlier is more important
    lex_order: Tuple[str, ...] = (
        "score_excess",
        "segment_length",
        "constraint_penalty",
        "boundary_discontinuity",
        "pca_reconstruction_error",
        "donor_distance",
    )


@dataclass
class CandidateEval:
    x_cf: Tensor
    score: float
    valid: bool
    replaced_segment: Tuple[int, int]
    donor_id: int
    donor_segment: Tuple[int, int]
    donor_distance: float
    constraint_feasible: bool
    hard_violations: Dict[str, float]
    soft_metrics: Dict[str, float]
    objectives: Dict[str, float]
    loss: Optional[float] = None


def _norm01(x: float, eps: float = 1e-9) -> float:
    # caller should feed roughly bounded numbers; this is a safe clamp helper
    if not np.isfinite(x):
        return 1.0
    return float(max(0.0, min(1.0, x + 0.0)))  # no scaling here; keep explicit


def _excess(score: float, tau: float) -> float:
    return float(max(0.0, score - tau))


def _constraint_penalty(hard: Dict[str, float], soft: Dict[str, float]) -> float:
    # hard already makes infeasible; still quantify
    hard_pen = float(sum(abs(v) for v in hard.values()))
    # soft penalty: sum of key realism metrics (you can tune downstream)
    soft_pen = 0.0
    for k in ("rate_excess", "boundary_discontinuity", "pca_reconstruction_error"):
        if k in soft and np.isfinite(soft[k]):
            soft_pen += float(soft[k])
    return hard_pen + soft_pen


# -------------------------------
# Core selector
# -------------------------------


class CandidateSelector:
    def __init__(self, cfg: Optional[SelectionConfig] = None):
        self.cfg = cfg or SelectionConfig()

    def evaluate_candidates(
        self,
        x: Tensor,  # (L,F)
        segment_candidates: List[Any],  # Part1 SegmentCandidate
        donor_matches: Dict[Tuple[int, int], Any],  # Part2 DonorMatchResult
        substitutor: Any,  # Part3 SegmentSubstitutor
        constraint_evaluator: Any,  # Part4 ConstraintEvaluator
        reconstruction_score: Callable[[Tensor], float],  # black-box score(x)->float
        normal_core: Tensor,  # (K,L,F) used only for donor segment extraction
    ) -> List[CandidateEval]:
        cfg = self.cfg
        evals: List[CandidateEval] = []

        seg_list = segment_candidates[: cfg.max_segments]

        for seg in seg_list:
            seg_key = (int(seg.start), int(seg.end))
            if seg_key not in donor_matches:
                continue

            donors = donor_matches[seg_key].donors[: cfg.max_donors_per_segment]

            for d in donors:
                # extract donor segment from NormalCore
                donor_seg = (
                    normal_core[d.donor_id, d.start : d.end]
                    .detach()
                    .to(x.device)
                    .float()
                )

                # substitute
                sub_res = substitutor.substitute(
                    x=x,
                    segment=seg_key,
                    donor_segment=donor_seg,
                    donor_id=int(d.donor_id),
                    donor_range=(int(d.start), int(d.end)),
                )

                x_cf = sub_res.x_cf

                # constraints
                cons = constraint_evaluator.evaluate(
                    x_orig=x,
                    x_cf=x_cf,
                    replaced_segment=seg_key,
                )

                # score
                s = float(reconstruction_score(x_cf))
                ex = _excess(s, cfg.tau)

                valid = bool(cons.feasible and (s <= cfg.tau))

                # objectives (raw, not normalized)
                obj = {
                    "score_excess": ex,
                    "segment_length": float(seg_key[1] - seg_key[0]),
                    "constraint_penalty": _constraint_penalty(
                        cons.hard_violations, cons.soft_metrics
                    ),
                    "boundary_discontinuity": float(
                        cons.soft_metrics.get("boundary_discontinuity", 0.0)
                    ),
                    "pca_reconstruction_error": float(
                        cons.soft_metrics.get("pca_reconstruction_error", 0.0)
                    ),
                    "donor_distance": float(d.distance),
                }

                evals.append(
                    CandidateEval(
                        x_cf=x_cf,
                        score=s,
                        valid=valid,
                        replaced_segment=seg_key,
                        donor_id=int(d.donor_id),
                        donor_segment=(int(d.start), int(d.end)),
                        donor_distance=float(d.distance),
                        constraint_feasible=bool(cons.feasible),
                        hard_violations=dict(cons.hard_violations),
                        soft_metrics=dict(cons.soft_metrics),
                        objectives=obj,
                        loss=None,
                    )
                )

        return evals

    def select_best(self, evals: List[CandidateEval]) -> Optional[CandidateEval]:
        cfg = self.cfg
        if not evals:
            return None

        # prioritize valid first
        valid = [e for e in evals if e.valid]
        pool = (
            valid if valid else evals
        )  # if none valid, still pick "least bad" (Part 6 handles failure)

        if cfg.mode == "single":
            # single loss = weighted sum, with explicit terms
            # IMPORTANT: we do NOT auto-normalize with dataset stats here.
            # Keep deterministic: user can add calibration later.
            for e in pool:
                seg_len = e.objectives["segment_length"]
                ex = e.objectives["score_excess"]
                cpen = e.objectives["constraint_penalty"]
                bd = e.objectives["boundary_discontinuity"]
                pca = e.objectives["pca_reconstruction_error"]
                dd = e.objectives["donor_distance"]

                # cheap scaling to avoid insane domination by a single term
                # (practical defaults; you can replace with quantile-based scaling later)
                seg_len_s = seg_len / max(
                    1.0, seg_len
                )  # -> 1.0 (keeps alpha meaningful but not dominating)
                ex_s = ex  # already in score units; beta sets importance
                cpen_s = cpen
                bd_s = bd
                pca_s = pca
                dd_s = dd

                e.loss = float(
                    cfg.alpha_len * seg_len_s
                    + cfg.beta_excess * ex_s
                    + cfg.gamma_constraints * cpen_s
                    + cfg.eta_boundary * bd_s
                    + cfg.zeta_pca * pca_s
                    + cfg.delta_normalcore_dist * dd_s
                )

            pool.sort(key=lambda z: (z.loss if z.loss is not None else float("inf")))
            return pool[0]

        if cfg.mode == "multi":
            # practical lexicographic ranking (traditional, stable, enforceable)
            order = cfg.lex_order

            def key_fn(e: CandidateEval):
                return tuple(float(e.objectives.get(k, 0.0)) for k in order)

            pool.sort(key=key_fn)
            return pool[0]

        raise ValueError(f"Unknown mode={cfg.mode}")

    def build_output(self, best: CandidateEval) -> Dict[str, Any]:
        return {
            "x_cf": best.x_cf,
            "score": float(best.score),
            "meta": {
                "replaced_segment": tuple(map(int, best.replaced_segment)),
                "segment_length": int(
                    best.replaced_segment[1] - best.replaced_segment[0]
                ),
                "donor_id": int(best.donor_id),
                "donor_segment": (
                    int(best.donor_id),
                    int(best.donor_segment[0]),
                    int(best.donor_segment[1]),
                ),
                "constraint_metrics": {
                    "constraint_feasible": bool(best.constraint_feasible),
                    "hard_violations": dict(best.hard_violations),
                    "soft_metrics": dict(best.soft_metrics),
                },
                "objectives": dict(best.objectives),
                "loss": None if best.loss is None else float(best.loss),
                "libraries_used": [
                    "torch",
                    "numpy",
                    "tslearn",
                    "scipy",
                    "scikit-learn",
                    "einops",
                    "dataclasses",
                ],
                "method": "comte_style",
            },
        }


# -------------------------------
# Example (glue-only demo)
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # fake x and NormalCore
    K, L, F = 8, 128, 6
    normal_core = torch.randn(K, L, F)
    x = torch.randn(L, F)

    # fake stubs for required components
    class _Seg:
        def __init__(self, s, t):
            self.start = s
            self.end = t

    segment_candidates = [_Seg(20, 36), _Seg(60, 84)]

    class _Donor:
        def __init__(self, donor_id, start, end, dist):
            self.donor_id = donor_id
            self.start = start
            self.end = end
            self.distance = dist

    class _DonorMatchResult:
        def __init__(self, seg, donors):
            self.segment = seg
            self.donors = donors

    donor_matches = {
        (20, 36): _DonorMatchResult(
            (20, 36), [_Donor(0, 10, 26, 1.2), _Donor(3, 50, 66, 1.5)]
        ),
        (60, 84): _DonorMatchResult((60, 84), [_Donor(2, 20, 44, 0.9)]),
    }

    # minimal substitutor + constraint evaluator (import your Part3/Part4 in real use)
    from dataclasses import dataclass

    @dataclass
    class _SubRes:
        x_cf: Tensor
        replaced_segment: Tuple[int, int]
        donor_id: int
        donor_segment: Tuple[int, int]
        boundary_applied: bool

    class _Sub:
        def substitute(self, x, segment, donor_segment, donor_id, donor_range):
            x_cf = x.clone()
            s, t = segment
            x_cf[s:t] = donor_segment
            return _SubRes(x_cf, segment, donor_id, donor_range, False)

    class _ConsRes:
        def __init__(self):
            self.feasible = True
            self.hard_violations = {}
            self.soft_metrics = {
                "boundary_discontinuity": 0.1,
                "pca_reconstruction_error": 0.2,
            }

    class _Cons:
        def evaluate(self, x_orig, x_cf, replaced_segment):
            return _ConsRes()

    def reconstruction_score(x_cf: Tensor) -> float:
        # fake score; in real use call your black-box detector
        return float(torch.mean((x_cf - 0.0) ** 2).item())

    selector = CandidateSelector(SelectionConfig(mode="multi", tau=1.0))
    evals = selector.evaluate_candidates(
        x=x,
        segment_candidates=segment_candidates,
        donor_matches=donor_matches,
        substitutor=_Sub(),
        constraint_evaluator=_Cons(),
        reconstruction_score=reconstruction_score,
        normal_core=normal_core,
    )
    best = selector.select_best(evals)
    if best is not None:
        out = selector.build_output(best)
        assert "x_cf" in out and "meta" in out
