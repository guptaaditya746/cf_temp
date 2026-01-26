# part6_final_assembly.py
# Generative Infilling Counterfactuals — Part 6: Final Assembly & Orchestration
#
# Responsibilities:
# - wire Parts 1–5 into a single generate() call
# - enforce strict execution order
# - guarantee REQUIRED OUTPUT FORMAT
# - NEVER return partial or silent results
#
# This is the only entry point users should call.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from methods.generative.candidate_selection import CandidateSelector, SelectionConfig
from methods.generative.constraint_evaluation import (
    ConstraintConfig,
    ConstraintEvaluator,
    ConstraintResult,
)
from methods.generative.failure_handling import FailureHandler, FailureReport
from methods.generative.infilling_engine import (
from methods.generative.infilling_engine import (
    InfillingEngine,
    InfillingConfig,
    InfillCandidate,
)
from methods.generative.constraint_evaluation import (
    ConstraintConfig,
    ConstraintEvaluator,
    ConstraintResult,
)
from methods.generative.candidate_selection import CandidateSelector, SelectionConfig
from methods.generative.failure_handling import FailureHandler, FailureReport



# Expected imports if packaged
# from part1_mask_strategy import MaskStrategy, MaskStrategyConfig
# from part2_infilling_engine import InfillingEngine, InfillingConfig, InfillCandidate
# from part3_constraint_evaluation import ConstraintEvaluator, ConstraintConfig, ConstraintResult
# from part4_candidate_selection import CandidateSelector, SelectionConfig
# from part5_failure_handling import FailureHandler, FailureReport
from methods.generative.mask_strategy import MaskStrategy, MaskStrategyConfig


# ----------------------------
# Output schema (MANDATORY)
# ----------------------------
@dataclass
class CounterfactualResult:
    x_cf: np.ndarray
    score: float
    meta: Dict[str, Any]


# ----------------------------
# Orchestrator
# ----------------------------
class GenerativeInfillingCounterfactual:
    """
    End-to-end counterfactual generator.

    Strict pipeline:
    1) Mask strategy
    2) Infilling
    3) Constraint evaluation
    4) Candidate selection
    5) Failure handling
    """

    def __init__(
        self,
        *,
        mask_strategy,
        infilling_engine,
        constraint_evaluator,
        candidate_selector,
        failure_handler,
    ):
        self.mask_strategy = mask_strategy
        self.infilling_engine = infilling_engine
        self.constraint_evaluator = constraint_evaluator
        self.candidate_selector = candidate_selector
        self.failure_handler = failure_handler

    def generate(
        self,
        *,
        x: np.ndarray,  # (L,F)
        reconstruction_error_t: np.ndarray,  # (L,)
        anomaly_score_fn: Callable[[np.ndarray], float],
        threshold: float,
        reconstruction_error_tf: Optional[np.ndarray] = None,  # (L,F)
        immutable_features: Optional[List[int]] = None,
    ) -> Optional[CounterfactualResult]:
        """
        Returns CounterfactualResult or None (with failure diagnostics emitted).
        """

        # ----------------------------
        # Input sanitation
        # ----------------------------
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"x must be (L,F), got {x.shape}")

        L, F = x.shape
        e_t = np.asarray(reconstruction_error_t, dtype=np.float32)
        if e_t.shape != (L,):
            raise ValueError("reconstruction_error_t must be (L,)")

        if reconstruction_error_tf is not None:
            e_tf = np.asarray(reconstruction_error_tf, dtype=np.float32)
            if e_tf.shape != (L, F):
                raise ValueError("reconstruction_error_tf must be (L,F)")
        else:
            e_tf = None

        imm = immutable_features or []

        # ----------------------------
        # PART 1 — Mask strategy
        # ----------------------------
        mask_specs = self.mask_strategy.propose(
            e_t=e_t,
            L=L,
            e_tf=e_tf,
            immutable_features=imm,
        )

        if not mask_specs:
            self._emit_failure(
                mask_specs=mask_specs,
            )
            return None

        # ----------------------------
        # PART 2 — Infilling
        # ----------------------------
        x_torch = torch.from_numpy(x)

        all_candidates = []
        for spec in mask_specs:
            cands = self.infilling_engine.generate(
                x=x_torch,
                mask_spec=spec,
                mask_builder=self.mask_strategy.build_mask,
                immutable_features=imm,
            )
            all_candidates.extend(cands)

        if not all_candidates:
            self._emit_failure(
                mask_specs=mask_specs,
                infill_candidates=[],
            )
            return None

        # ----------------------------
        # PART 3 — Constraint evaluation
        # ----------------------------
        constraint_results = []
        for cand in all_candidates:
            cres = self.constraint_evaluator.evaluate(
                x_orig=x,
                x_cf=cand.x_cf.detach().cpu().numpy(),
                mask=cand.mask.detach().cpu().numpy(),
            )
            constraint_results.append(cres)

        # ----------------------------
        # PART 4 — Candidate selection
        # ----------------------------
        # Update selector threshold dynamically
        self.candidate_selector.cfg.threshold = float(threshold)
        self.candidate_selector.score_fn = anomaly_score_fn

        selection = self.candidate_selector.select(
            candidates=all_candidates,
            constraint_results=constraint_results,
        )

        if selection is not None:
            return CounterfactualResult(
                x_cf=selection.x_cf,
                score=selection.score,
                meta=selection.meta,
            )

        # ----------------------------
        # PART 5 — Failure handling
        # ----------------------------
        scores = [
            float(anomaly_score_fn(c.x_cf.detach().cpu().numpy()))
            for c in all_candidates
        ]

        self._emit_failure(
            mask_specs=mask_specs,
            infill_candidates=all_candidates,
            constraint_results=constraint_results,
            scores=scores,
            threshold=threshold,
            x_orig=x,
        )

        return None

    # ----------------------------
    # Failure emission
    # ----------------------------
    def _emit_failure(self, **kwargs) -> None:
        report = self.failure_handler.analyze(**kwargs)
        # Hard rule: no silent failure
        # In production this should be logged, raised, or returned upstream
        print("COUNTERFACTUAL FAILURE")
        print("Type:", report.failure_type)
        print("Message:", report.message)
        print("Diagnostics:", report.diagnostics)
        print("Suggested action:", report.suggested_action)


# ----------------------------
# Minimal integration test
# ----------------------------
if __name__ == "__main__":
    # NOTE: This assumes Parts 1–5 are instantiated properly.
    # This is only a wiring sanity check.

    def dummy_score(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x)))

    # Fake objects (replace with real ones in actual usage)
    class _Dummy:
        pass

    print("Part 6 loaded. Ready for full pipeline integration.")
