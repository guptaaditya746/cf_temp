# part5_failure_handling.py
# Generative Infilling Counterfactuals — Part 5: Failure Handling
#
# Responsibilities:
# - detect explicit failure modes
# - classify why no counterfactual was produced
# - NEVER fail silently
# - return structured diagnostics usable by downstream systems / logs
#
# This module does NOT retry generation or selection.
# It explains WHY the pipeline failed so strategy can be adjusted.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# ----------------------------
# Failure Types (Explicit)
# ----------------------------
FAILURE_NO_MASKS = "no_masks_generated"
FAILURE_NO_INFILL = "no_infilling_candidates"
FAILURE_ALL_HARD_CONSTRAINTS = "all_failed_hard_constraints"
FAILURE_ALL_SOFT_IMPLAUSIBLE = "all_soft_implausible"
FAILURE_THRESHOLD_TOO_STRICT = "threshold_too_strict"
FAILURE_FLATLINE_INFILL = "flatline_infilling"
FAILURE_OVER_MASKING = "over_masking"
FAILURE_REGIME_TELEPORT = "regime_teleportation"
FAILURE_UNKNOWN = "unknown_failure"


# ----------------------------
# Output
# ----------------------------
@dataclass
class FailureReport:
    failure_type: str
    message: str
    diagnostics: Dict[str, Any]
    suggested_action: str


# ----------------------------
# Handler
# ----------------------------
class FailureHandler:
    """
    Inspects intermediate artifacts and determines why no valid CF was produced.
    This is deliberately conservative and explicit.
    """

    def analyze(
        self,
        *,
        mask_specs: Optional[List[Any]] = None,
        infill_candidates: Optional[List[Any]] = None,
        constraint_results: Optional[List[Any]] = None,
        scores: Optional[List[float]] = None,
        threshold: Optional[float] = None,
        x_orig: Optional[np.ndarray] = None,
    ) -> FailureReport:
        # 1) No masks at all
        if not mask_specs:
            return FailureReport(
                failure_type=FAILURE_NO_MASKS,
                message="No anomalous segments could be identified for masking.",
                diagnostics={},
                suggested_action="Lower segment threshold, increase top_k, or reduce min_len.",
            )

        # 2) Masks exist but no infilling output
        if not infill_candidates:
            return FailureReport(
                failure_type=FAILURE_NO_INFILL,
                message="Infilling engine produced no candidates.",
                diagnostics={"num_masks": len(mask_specs)},
                suggested_action="Increase n_samples or inspect infiller model compatibility.",
            )

        # 3) Hard constraint rejection
        if constraint_results and all(not c.feasible for c in constraint_results):
            violations = {}
            for c in constraint_results:
                for k in c.hard_violations:
                    violations[k] = violations.get(k, 0) + 1

            return FailureReport(
                failure_type=FAILURE_ALL_HARD_CONSTRAINTS,
                message="All candidates violated hard constraints.",
                diagnostics={"violations": violations},
                suggested_action="Relax hard bounds or reduce mask size.",
            )

        # 4) Soft implausibility (all penalties extreme)
        if constraint_results and all(
            c.feasible and c.total_soft_penalty > 0 for c in constraint_results
        ):
            penalties = [c.total_soft_penalty for c in constraint_results]
            return FailureReport(
                failure_type=FAILURE_ALL_SOFT_IMPLAUSIBLE,
                message="All candidates were implausible under soft constraints.",
                diagnostics={
                    "min_penalty": float(np.min(penalties)),
                    "median_penalty": float(np.median(penalties)),
                },
                suggested_action="Reduce curvature/spectral weights or allow larger edits.",
            )

        # 5) Threshold too strict
        if scores is not None and threshold is not None:
            if all(s > threshold for s in scores):
                return FailureReport(
                    failure_type=FAILURE_THRESHOLD_TOO_STRICT,
                    message="No candidate achieved reconstruction score below threshold.",
                    diagnostics={
                        "best_score": float(np.min(scores)),
                        "threshold": float(threshold),
                    },
                    suggested_action="Relax τ slightly or expand mask size.",
                )

        # 6) Flatline infilling detection
        if x_orig is not None and infill_candidates:
            flat = 0
            for c in infill_candidates:
                x_cf = c.x_cf.detach().cpu().numpy()
                if np.std(x_cf) < 1e-4:
                    flat += 1
            if flat == len(infill_candidates):
                return FailureReport(
                    failure_type=FAILURE_FLATLINE_INFILL,
                    message="All infilled candidates collapsed to near-constant signals.",
                    diagnostics={"std_threshold": 1e-4},
                    suggested_action="Change mask token or use a stronger infiller model.",
                )

        # 7) Over-masking detection
        if infill_candidates:
            ratios = [
                c.mask_size / (c.x_cf.shape[0] * c.x_cf.shape[1])
                for c in infill_candidates
            ]
            if max(ratios) > 0.5:
                return FailureReport(
                    failure_type=FAILURE_OVER_MASKING,
                    message="Counterfactual requires masking most of the window.",
                    diagnostics={"max_mask_ratio": float(max(ratios))},
                    suggested_action="Reject explanation or cap maximum mask size.",
                )

        # 8) Regime teleportation (large global shift)
        if x_orig is not None and infill_candidates:
            shifts = []
            for c in infill_candidates:
                x_cf = c.x_cf.detach().cpu().numpy()
                shifts.append(float(np.mean(np.abs(x_cf - x_orig))))
            if min(shifts) > 3.0 * np.std(x_orig):
                return FailureReport(
                    failure_type=FAILURE_REGIME_TELEPORT,
                    message="Infilling causes global regime shift.",
                    diagnostics={"min_mean_shift": min(shifts)},
                    suggested_action="Constrain infilling context window or penalize DTW more strongly.",
                )

        # Fallback
        return FailureReport(
            failure_type=FAILURE_UNKNOWN,
            message="Counterfactual generation failed for unspecified reasons.",
            diagnostics={},
            suggested_action="Inspect intermediate artifacts manually.",
        )


# ----------------------------
# Minimal self-test
# ----------------------------
if __name__ == "__main__":
    fh = FailureHandler()

    rep = fh.analyze(mask_specs=[])
    print(rep.failure_type, "->", rep.message)

    rep = fh.analyze(
        mask_specs=[1, 2],
        infill_candidates=[],
    )
    print(rep.failure_type, "->", rep.message)
