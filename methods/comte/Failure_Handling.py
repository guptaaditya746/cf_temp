# Part 6 — Failure Handling & Reporting
# - detect explicit failure modes
# - NEVER fail silently
# - return structured diagnostics when no valid CF exists
#
# This part does NOT try to "fix" failures.
# It reports them cleanly so downstream code or users can decide what to do.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# -------------------------------
# Failure containers
# -------------------------------


@dataclass
class FailureReport:
    reason: str
    details: Dict[str, Any]
    method: str = "comte_style"


# -------------------------------
# Failure analyzer
# -------------------------------


class FailureHandler:
    """
    Interprets the evaluation results from Part 5 and reports
    WHY a counterfactual could not be produced.
    """

    def analyze(
        self,
        evals: List[Any],  # List[CandidateEval] from Part 5
        tau: float,
    ) -> Optional[FailureReport]:
        """
        Returns FailureReport if CF failed, otherwise None.
        """

        if not evals:
            return FailureReport(
                reason="no_candidates_generated",
                details={
                    "message": "No segment–donor combinations were evaluated.",
                    "likely_causes": [
                        "NormalCore too small",
                        "Segment generation too strict",
                        "Donor matching failed",
                    ],
                },
            )

        # ---------- VALIDITY CHECK ----------
        valid = [e for e in evals if e.valid]
        if valid:
            return None  # success path; no failure

        # ---------- FAILURE MODE ANALYSIS ----------
        reasons: Dict[str, int] = {
            "threshold_not_reached": 0,
            "constraint_violation": 0,
            "unrealistic_donor": 0,
            "boundary_discontinuity": 0,
        }

        worst_excess = 0.0

        for e in evals:
            # score failure
            excess = max(0.0, e.score - tau)
            worst_excess = max(worst_excess, excess)

            if excess > 0:
                reasons["threshold_not_reached"] += 1

            if not e.constraint_feasible:
                reasons["constraint_violation"] += 1

            if e.objectives.get("donor_distance", 0.0) > 0:
                reasons["unrealistic_donor"] += 1

            if e.objectives.get("boundary_discontinuity", 0.0) > 0:
                reasons["boundary_discontinuity"] += 1

        # dominant reason
        dominant = max(reasons.items(), key=lambda z: z[1])[0]

        # ---------- STRUCTURED REPORT ----------
        return FailureReport(
            reason=dominant,
            details={
                "evaluated_candidates": len(evals),
                "worst_score_excess": float(worst_excess),
                "reason_counts": reasons,
                "interpretation": self._interpret(dominant),
                "recommended_actions": self._recommend(dominant),
            },
        )

    # ---------------------------
    # Interpretation helpers
    # ---------------------------

    def _interpret(self, reason: str) -> str:
        if reason == "threshold_not_reached":
            return (
                "All candidate counterfactuals reduced the anomaly score, "
                "but none crossed the decision threshold."
            )
        if reason == "constraint_violation":
            return (
                "Counterfactuals reached the threshold but violated "
                "hard realism constraints."
            )
        if reason == "unrealistic_donor":
            return "NormalCore segments were too dissimilar from the anomalous segment."
        if reason == "boundary_discontinuity":
            return "Segment substitutions introduced sharp temporal discontinuities."
        return "Unknown failure mode."

    def _recommend(self, reason: str) -> List[str]:
        if reason == "threshold_not_reached":
            return [
                "Relax τ slightly and re-run",
                "Allow longer segment replacements",
                "Increase number of donor segments",
            ]
        if reason == "constraint_violation":
            return [
                "Relax soft constraints",
                "Review immutable sensor list",
                "Inspect sensor bounds",
            ]
        if reason == "unrealistic_donor":
            return [
                "Expand NormalCore",
                "Switch similarity metric (DTW ↔ Euclidean)",
                "Allow regime-specific donor selection",
            ]
        if reason == "boundary_discontinuity":
            return [
                "Increase boundary smoothing",
                "Allow slightly longer segments",
            ]
        return ["Inspect logs and diagnostics manually"]


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Fake minimal CandidateEval-like object
    class _Eval:
        def __init__(self, valid, score, constraint_feasible, objectives):
            self.valid = valid
            self.score = score
            self.constraint_feasible = constraint_feasible
            self.objectives = objectives

    evals = [
        _Eval(False, 3.2, True, {"donor_distance": 1.5, "boundary_discontinuity": 0.2}),
        _Eval(
            False, 2.8, False, {"donor_distance": 1.1, "boundary_discontinuity": 0.0}
        ),
    ]
    handler = FailureHandler()
    report = handler.analyze(evals, tau=1.0)

    assert report is not None
    assert "reason" in asdict(report)
