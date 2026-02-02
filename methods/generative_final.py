# part6_final_assembly.py
# Generative Infilling Counterfactuals — Part 6: Final Assembly & Orchestration
#
# Responsibilities:
# - Wire Parts 1–5 into a single generate() call
# - Enforce strict execution order
# - Guarantee REQUIRED OUTPUT FORMAT
# - NEVER return partial or silent results

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

# =============================================================================
# OPTIONAL IMPORTS (For type hinting context if packages exist)
# In a real package, you would import these from their respective modules.
# =============================================================================
# from part1_mask_strategy import MaskStrategy
# from part2_infilling_engine import InfillingEngine, InfillCandidate
# from part3_constraint_evaluation import ConstraintEvaluator, ConstraintResult
# from part4_candidate_selection import CandidateSelector
# from part5_failure_handling import FailureHandler
from methods.generative.constraint_evaluation import ConstraintResult


# -----------------------------------------------------------------------------
# 1. Output Schema (MANDATORY)
# -----------------------------------------------------------------------------
@dataclass
class CounterfactualResult:
    """
    Standardized output for a successful counterfactual generation.
    """

    xcf: np.ndarray  # The final counterfactual window (L, F)
    score: float  # The anomaly score of x_cf (should be < threshold)
    meta: Dict[str, Any]  # Provenance data (method used, mask ratio, etc.)


# -----------------------------------------------------------------------------
# 2. Orchestrator
# -----------------------------------------------------------------------------
class GenerativeInfillingCounterfactual:
    """
    End-to-end counterfactual generator.

    This class owns the "life cycle" of a counterfactual request.
    It does not contain logic itself; it coordinates the logic of Parts 1-5.

    Strict Pipeline:
    1. Mask Strategy:        Decide what to hide.
    2. Infilling Engine:     Hallucinate plausible replacements.
    3. Constraint Evaluator: Check physics/business rules.
    4. Candidate Selector:   Pick the best valid option.
    5. Failure Handling:     Diagnose why if no option is found.
    """

    def __init__(
        self,
        *,
        mask_strategy,  # Part 1 instance
        infilling_engine,  # Part 2 instance
        constraint_evaluator,  # Part 3 instance
        candidate_selector,  # Part 4 instance
        failure_handler,  # Part 5 instance
    ):
        self.mask_strategy = mask_strategy
        self.infilling_engine = infilling_engine
        self.constraint_evaluator = constraint_evaluator
        self.candidate_selector = candidate_selector
        self.failure_handler = failure_handler

    def generate(
        self,
        *,
        x: np.ndarray,  # (L, F) Input window
        reconstruction_error_t: np.ndarray,  # (L,) Time-wise error
        anomaly_score_fn: Callable[[np.ndarray], float],  # Function to score candidates
        threshold: float,  # Target score to beat
        reconstruction_error_tf: Optional[np.ndarray] = None,  # (L, F) detailed error
        immutable_features: Optional[List[int]] = None,
    ) -> Optional[CounterfactualResult]:
        """
        Main entry point. Returns CounterfactualResult or None (with diagnostics).
        """

        # ---------------------------------------------------------------------
        # A. Input Sanitation
        # ---------------------------------------------------------------------
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Input x must be (L, F), got {x.shape}")

        L, F = x.shape
        e_t = np.asarray(reconstruction_error_t, dtype=np.float32)

        if e_t.shape != (L,):
            raise ValueError(f"reconstruction_error_t must be ({L},), got {e_t.shape}")

        e_tf = None
        if reconstruction_error_tf is not None:
            e_tf = np.asarray(reconstruction_error_tf, dtype=np.float32)
            if e_tf.shape != (L, F):
                raise ValueError(f"reconstruction_error_tf must be ({L}, {F})")

        imm = immutable_features or []

        # ---------------------------------------------------------------------
        # B. PART 1: Mask Strategy
        # ---------------------------------------------------------------------
        # "Where is the problem?"
        mask_specs = self.mask_strategy.propose(
            e_t=e_t, L=L, e_tf=e_tf, immutable_features=imm
        )

        if not mask_specs:
            self._emit_failure(
                stage="Mask Strategy",
                message="No mask specifications proposed (data might be normal?).",
                context={"x": x},
            )
            return None

        # ---------------------------------------------------------------------
        # C. PART 2: Infilling (Generative Step)
        # ---------------------------------------------------------------------
        # "What could replace the problem areas?"
        x_torch = torch.from_numpy(x)
        all_candidates = []

        # We iterate over all proposed masking strategies (e.g., Sparse vs Dense)
        for spec in mask_specs:
            # The engine calls the Generative Model (VAE/Transformer) here
            cands = self.infilling_engine.generate(
                x=x_torch,
                mask_spec=spec,
                mask_builder=self.mask_strategy.build_mask,
                immutable_features=imm,
            )
            all_candidates.extend(cands)

        if not all_candidates:
            self._emit_failure(
                stage="Infilling",
                message="Infilling engine produced zero candidates.",
                context={"mask_specs": mask_specs},
            )
            return None

        # ---------------------------------------------------------------------
        # D. PART 3: Constraint Evaluation
        # ---------------------------------------------------------------------
        # "Are these replacements physically possible?"
        # D. PART 3: Constraint Evaluation
        # D. PART 3: Constraint Evaluation
        constraint_results = []
        for cand in all_candidates:
            # Ensure proper numpy conversion
            x_cf_np = cand.x_cf.detach().cpu().numpy().astype(np.float32)

            # CRITICAL: Ensure mask is boolean numpy array
            mask_np = cand.mask.detach().cpu().numpy()
            if mask_np.dtype != bool:
                mask_np = mask_np.astype(bool)

            # Validate shapes before evaluation
            if x_cf_np.shape != x.shape:
                print(f"Shape mismatch: x_cf {x_cf_np.shape} vs x {x.shape}")
                continue

            if mask_np.shape != x.shape:
                print(f"Mask shape mismatch: {mask_np.shape} vs {x.shape}")
                continue

            try:
                cres = self.constraint_evaluator.evaluate(
                    x_orig=x, x_cf=x_cf_np, mask=mask_np
                )
                constraint_results.append(cres)
            except Exception as e:
                print(f"Constraint evaluation failed for candidate: {e}")
                # Create a failure result
                cres = ConstraintResult(
                    feasible=False,
                    hard_violations={"evaluation_error": str(e)},
                    soft_metrics={},
                    total_soft_penalty=float("inf"),
                )
                constraint_results.append(cres)

        # ---------------------------------------------------------------------
        # E. PART 4: Candidate Selection
        # ---------------------------------------------------------------------
        # "Which replacement is the best valid explanation?"

        # Inject dynamic runtime requirements into selector
        self.candidate_selector.cfg.threshold = float(threshold)
        self.candidate_selector.score_fn = anomaly_score_fn

        selection = self.candidate_selector.select(
            candidates=all_candidates,
            constraint_results=constraint_results,
        )

        # SUCCESS: Return standardized result
        # In GenerativeInfillingCounterfactual.generate(), before the return:
        if selection is not None:
            # Debug: Print what selection contains

            return CounterfactualResult(
                xcf=selection.x_cf,
                score=selection.score,
                meta=selection.meta,
            )

        # ---------------------------------------------------------------------
        # F. PART 5: Failure Handling
        # ---------------------------------------------------------------------
        # "Why did we fail?"

        # Calculate scores for all failed candidates to help diagnosis
        candidate_scores = []
        for cand in all_candidates:
            x_cf_np = cand.x_cf.detach().cpu().numpy()
            s = float(anomaly_score_fn(x_cf_np))
            candidate_scores.append(s)

        self._emit_failure(
            mask_specs=mask_specs,
            infill_candidates=all_candidates,
            constraint_results=constraint_results,
            scores=candidate_scores,
            threshold=threshold,
            x_orig=x,
            # stage="Selection",
        )

        return None

    def _emit_failure(self, **kwargs) -> None:
        """
        Delegates to FailureHandler and prints the report.
        In a production API, this might log to a file or raise an Exception.
        """
        stage = kwargs.pop("stage", "Unknown")
        report = self.failure_handler.analyze(**kwargs)

        print("\n" + "!" * 60)
        print("FAILED TO GENERATE COUNTERFACTUAL")
        print(f"stage: {stage}")
        print(f"Type:       {report.failure_type}")
        print(f"Reason:     {report.message}")
        print("-" * 30)
        print("Diagnostics:")
        for k, v in report.diagnostics.items():
            print(f"  - {k}: {v}")
        print("-" * 30)
        print(f"Suggestion: {report.suggested_action}")
        print("!" * 60 + "\n")


# -----------------------------------------------------------------------------
# Usage Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Initializing Counterfactual Pipeline Components...")

    # 1. Instantiate the classes from Parts 1-5 (Mocking them here for the script to run standalone)
    class MockStrategy:
        def propose(self, **kwargs):
            return ["spec1"]

        def build_mask(self, **kwargs):
            return torch.ones(10, 6)  # Dummy mask

    class MockInfilling:
        def generate(self, **kwargs):
            # Returns a dummy candidate
            from types import SimpleNamespace

            return [SimpleNamespace(x_cf=torch.randn(10, 6), mask=torch.ones(10, 6))]

    class MockConstraints:
        def evaluate(self, **kwargs):
            return "Valid"

    class MockSelector:
        def __init__(self):
            self.cfg = type("cfg", (), {"threshold": 0.5})

        def select(self, **kwargs):
            # Simulate success
            from types import SimpleNamespace

            return SimpleNamespace(x_cf=np.random.randn(10, 6), score=0.1, meta={})

    class MockFailure:
        def analyze(self, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(
                failure_type="Simulated",
                message="Test",
                diagnostics={},
                suggested_action="None",
            )

    # 2. Assembly
    pipeline = GenerativeInfillingCounterfactual(
        mask_strategy=MockStrategy(),
        infilling_engine=MockInfilling(),
        constraint_evaluator=MockConstraints(),
        candidate_selector=MockSelector(),
        failure_handler=MockFailure(),
    )

    print("Pipeline Assembled. Ready for generate().")
