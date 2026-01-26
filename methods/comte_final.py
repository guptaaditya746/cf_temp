# part6_final_assembly.py
# Final Assembly & Metadata Script
# - wires Part 1 -> Part 6 into a single callable pipeline
# - produces FINAL output format exactly as specified
# - handles success and failure explicitly

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from tqdm import tqdm

Tensor = torch.Tensor


class CoMTEReconstructionCF:
    """
    End-to-end CoMTE-style counterfactual generator
    for reconstruction-based time-series anomaly detection.
    """

    def __init__(
        self,
        segment_generator,  # Part 1 (already bound to reconstructor)
        donor_matcher,  # Part 2
        substitutor,  # Part 3
        constraint_evaluator,  # Part 4
        selector,  # Part 5
        failure_handler,  # Part 6
        reconstruction_score: Callable[[Tensor], float],
        tau: float,
        normal_core: Tensor,  # (K,L,F)
    ):
        self.segment_generator = segment_generator
        self.donor_matcher = donor_matcher
        self.substitutor = substitutor
        self.constraint_evaluator = constraint_evaluator
        self.selector = selector
        self.failure_handler = failure_handler
        self.reconstruction_score = reconstruction_score
        self.tau = float(tau)
        self.normal_core = normal_core

    # -------------------------------------------------

    def generate(self, x: Tensor) -> Optional[Dict[str, Any]]:
        """
        Returns:
          - Final CF dict, OR
          - Failure report dict
        """

        # ---------- PART 1 ----------
        seg_out = self.segment_generator.generate(x)
        segment_candidates = seg_out.candidates

        if not segment_candidates:
            return self._fail("no_segments", "No anomalous segments detected.")

        # ---------- PART 2 ----------
        donor_matches = self.donor_matcher.match(
            x=x,
            segment_candidates=segment_candidates,
        )

        if not donor_matches:
            return self._fail(
                "no_donors",
                "No matching donor segments found in NormalCore."
                "No matching donor segments found in NormalCore.",
            )

        # ---------- PART 5 (includes 3 & 4) ----------
        print(f"Evaluation started for {len(segment_candidates)} candidates...")
        evals = self.selector.evaluate_candidates(
            x=x,
            segment_candidates=segment_candidates,
            donor_matches=donor_matches,
            substitutor=self.substitutor,
            constraint_evaluator=self.constraint_evaluator,
            reconstruction_score=self.reconstruction_score,
            normal_core=self.normal_core,
        )

        best = self.selector.select_best(evals)

        if best is None or not best.valid:
            report = self.failure_handler.analyze(evals, tau=self.tau)
            return report.__dict__ if report else None

        return self.selector.build_output(best)

    # -------------------------------------------------

    def _fail(self, reason: str, message: str) -> Dict[str, Any]:
        return {
            "failure": {
                "reason": reason,
                "message": message,
                "method": "comte_style",
            }
        }
