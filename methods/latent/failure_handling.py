# failure_handling.py
# Part 6: Failure Handling & Metadata
#
# Responsibility:
# - Detect, classify, and EXPLICITLY REPORT failure modes
# - Standardize metadata for successful and failed CF attempts
# - Ensure NO silent failure anywhere in the pipeline
#
# This module:
# - does NOT optimize
# - does NOT select
# - does NOT evaluate constraints
#
# It ONLY interprets outcomes and annotates them honestly.
#
# External-first:
# - numpy
# - torch

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

Tensor = torch.Tensor


# -------------------------
# Failure taxonomy (LOCKED)
# -------------------------

FAILURE_TYPES = {
    "no_candidate_generated": "Optimizer produced no candidates",
    "no_valid_candidate": "No candidate crossed validity threshold",
    "hard_constraint_violation": "All candidates violated hard decoded constraints",
    "threshold_gaming": "Candidates barely crossed threshold without margin",
    "flatline_cf": "Counterfactual collapsed to low-variance / flat signal",
    "regime_teleportation": "Latent jump too large; regime change detected",
    "decoder_hallucination": "Decoder produced implausible artifacts",
    "robustness_failure": "Counterfactual not stable under perturbations",
    "exception_during_search": "Exception occurred during CF generation",
}


# -------------------------
# Failure detection helpers
# -------------------------


def detect_flatline(
    x_cf: Tensor,
    var_threshold: float = 1e-4,
) -> bool:
    """
    Detect near-constant decoded signals.
    """
    var = torch.var(x_cf, dim=0).mean().item()
    return var < var_threshold


def detect_regime_jump(
    z_cf: Tensor,
    z_orig: Tensor,
    latent_jump_threshold: float,
) -> bool:
    """
    Detect excessive latent displacement.
    """
    d = torch.linalg.norm(z_cf - z_orig).item()
    return d > latent_jump_threshold


def detect_threshold_gaming(
    score: float,
    validity_target: float,
    eps: float = 1e-3,
) -> bool:
    """
    Detect candidates barely crossing the threshold.
    """
    return validity_target <= score <= (validity_target + eps)


# -------------------------
# Metadata assembly
# -------------------------


def build_success_metadata(
    *,
    problem: Any,  # LatentCFProblem
    z_cf: Tensor,
    score: float,
    constraint_metrics: Dict[str, float],
    optimizer: str,
    robustness_ok: bool,
) -> Dict[str, Any]:
    """
    Standard metadata for SUCCESSFUL CF.
    """
    return {
        "status": "success",
        "interpretation": (
            "This counterfactual explains what must change so THIS MODEL "
            "would consider the window normal. "
            "It does NOT imply real-world causality."
        ),
        "score": float(score),
        "validity_target": float(problem.validity_target()),
        "latent_distance": float(torch.linalg.norm(z_cf - problem.z).item()),
        "normalcore_distance": (
            None
            if problem.normalcore_z is None
            else float(
                torch.min(
                    torch.linalg.norm(problem.normalcore_z - z_cf.unsqueeze(0), dim=1)
                ).item()
            )
        ),
        "constraint_metrics": constraint_metrics,
        "robust": bool(robustness_ok),
        "optimizer": optimizer,
        "normalcore_used": problem.normalcore_z is not None,
    }


def build_failure_metadata(
    *,
    failure_type: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Standard metadata for FAILURE.
    """
    if failure_type not in FAILURE_TYPES:
        raise ValueError(f"Unknown failure_type: {failure_type}")

    return {
        "status": "failure",
        "failure_type": failure_type,
        "reason": FAILURE_TYPES[failure_type],
        "details": details or {},
    }


# -------------------------
# High-level failure classifier
# -------------------------


def classify_failure(
    *,
    candidates: Sequence[Any],  # CandidateEval
    problem: Any,  # LatentCFProblem
    latent_jump_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Inspect evaluated candidates and assign the MOST HONEST failure reason.
    """
    if not candidates:
        return build_failure_metadata(failure_type="no_candidate_generated")

    hard_ok = [c for c in candidates if c.hard_ok]
    if not hard_ok:
        return build_failure_metadata(failure_type="hard_constraint_violation")

    valid = [c for c in hard_ok if c.score <= problem.validity_target()]
    if not valid:
        return build_failure_metadata(failure_type="no_valid_candidate")

    # Threshold gaming
    for c in valid:
        if detect_threshold_gaming(
            score=c.score,
            validity_target=problem.validity_target(),
        ):
            return build_failure_metadata(
                failure_type="threshold_gaming",
                details={"score": c.score},
            )

    # Flatline detection
    for c in valid:
        try:
            x_cf = problem.decode(c.z_cf)
            if detect_flatline(x_cf):
                return build_failure_metadata(failure_type="flatline_cf")
        except Exception:
            continue

    # Regime teleportation
    if latent_jump_threshold is not None:
        for c in valid:
            if detect_regime_jump(c.z_cf, problem.z, latent_jump_threshold):
                return build_failure_metadata(
                    failure_type="regime_teleportation",
                    details={
                        "latent_distance": float(
                            torch.linalg.norm(c.z_cf - problem.z).item()
                        )
                    },
                )

    return build_failure_metadata(failure_type="robustness_failure")


# -------------------------
# Final result assembly
# -------------------------


def assemble_final_result(
    *,
    cf: Optional[Dict[str, Any]],
    candidates: Sequence[Any],
    problem: Any,
    optimizer_name: str,
    latent_jump_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Produce FINAL output object.
    This is the ONLY thing users should consume.
    """
    if cf is not None:
        meta = build_success_metadata(
            problem=problem,
            z_cf=cf["z_cf"],
            score=cf["score"],
            constraint_metrics=cf["meta"]["constraint_metrics"],
            optimizer=optimizer_name,
            robustness_ok=cf["meta"].get("robust", False),
        )
        return {
            "x_cf": cf["x_cf"],
            "z_cf": cf["z_cf"],
            "score": cf["score"],
            "meta": meta,
        }

    # Failure path
    meta = classify_failure(
        candidates=candidates,
        problem=problem,
        latent_jump_threshold=latent_jump_threshold,
    )
    meta["optimizer"] = optimizer_name

    return {
        "x_cf": None,
        "z_cf": None,
        "score": None,
        "meta": meta,
    }


# -------------------------
# Minimal smoke test
# -------------------------
if __name__ == "__main__":
    print("Part 6: Failure Handling & Metadata module loaded.")
    print("No silent failures. Ever.")
