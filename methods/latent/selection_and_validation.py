# selection_and_validation.py
# Part 5: Final Selection & Validation
#
# Responsibility:
# - Select final counterfactual(s) from optimization outputs
# - Enforce STRICT acceptance policy (no shortcuts)
# - Perform robustness / sanity validation
#
# This module:
# - does NOT optimize
# - does NOT define objectives
# - does NOT evaluate constraints (delegates to Part 2)
#
# External-first:
# - numpy
# - torch

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Sequence

import numpy as np
import torch

Tensor = torch.Tensor


# -------------------------
# Acceptance configuration
# -------------------------


@dataclass
class SelectionConfig:
    """
    Controls how final CFs are selected.
    """

    require_validity: bool = True
    require_margin: bool = True
    margin_eps: float = 0.0  # additional margin beyond problem.eps_validity

    max_constraint_penalty: Optional[float] = None
    max_latent_distance: Optional[float] = None

    # Robustness checks
    robustness_trials: int = 8
    robustness_sigma: float = 0.05
    robustness_valid_frac: float = (
        0.75  # fraction of perturbed samples that must remain valid
    )

    # Selection preference
    prefer_min_latent_distance: bool = True
    prefer_lower_constraint_penalty: bool = True


# -------------------------
# Internal helpers
# -------------------------


def _perturb_latent(
    z: Tensor,
    sigma: float,
    rng: np.random.Generator,
) -> Tensor:
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=z.shape).astype(np.float32)
    ).to(z.device, z.dtype)
    return z + noise


def _robustness_check(
    *,
    problem: Any,  # LatentCFProblem
    z_cf: Tensor,
    x_orig: Tensor,
    constraint_eval: Any,  # DecodedConstraintEvaluator
    cfg: SelectionConfig,
    rng: np.random.Generator,
) -> bool:
    """
    Check that small perturbations around z_cf remain valid.
    """
    valid = 0
    trials = int(cfg.robustness_trials)
    if trials <= 0:
        return True

    target = problem.validity_target()

    for _ in range(trials):
        z_p = _perturb_latent(z_cf, cfg.robustness_sigma, rng)
        z_p = problem.apply_search_space(z_p)

        try:
            x_p = problem.decode(z_p)
            cres = constraint_eval.evaluate(x_p, x_orig=x_orig)
            if not cres.hard_ok:
                continue
            s = float(problem.score_fn(x_p))
            if s <= target:
                valid += 1
        except Exception:
            continue

    return (valid / trials) >= cfg.robustness_valid_frac


# -------------------------
# Selection logic
# -------------------------


def select_from_candidates(
    *,
    candidates: Sequence[Any],  # CandidateEval
    problem: Any,  # LatentCFProblem
    constraint_eval: Any,  # DecodedConstraintEvaluator
    x_orig: Tensor,
    cfg: SelectionConfig,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select a single final CF from a list of evaluated candidates
    (CMA-ES history or NSGA-II Pareto set).

    Returns:
        dict with x_cf, z_cf, score, meta
        or None if no acceptable CF exists.
    """
    if not candidates:
        return None

    rng = rng or np.random.default_rng()
    target = problem.validity_target() - float(cfg.margin_eps)

    accepted = []

    # -------------------------
    # Hard filtering
    # -------------------------
    for ce in candidates:
        if not ce.hard_ok:
            continue

        if cfg.require_validity and ce.score > target:
            continue

        if cfg.max_constraint_penalty is not None:
            if ce.constraint_penalty > cfg.max_constraint_penalty:
                continue

        if cfg.max_latent_distance is not None:
            d = torch.linalg.norm(ce.z_cf - problem.z).item()
            if d > cfg.max_latent_distance:
                continue

        accepted.append(ce)

    if not accepted:
        return None

    # -------------------------
    # Ranking
    # -------------------------
    def rank_key(ce):
        keys = []
        if cfg.prefer_min_latent_distance:
            keys.append(torch.linalg.norm(ce.z_cf - problem.z).item())
        if cfg.prefer_lower_constraint_penalty:
            keys.append(ce.constraint_penalty)
        return tuple(keys)

    accepted.sort(key=rank_key)

    # -------------------------
    # Robustness validation
    # -------------------------
    for ce in accepted:
        ok = _robustness_check(
            problem=problem,
            z_cf=ce.z_cf,
            x_orig=x_orig,
            constraint_eval=constraint_eval,
            cfg=cfg,
            rng=rng,
        )
        if not ok:
            continue

        # Decode final CF
        x_cf = problem.decode(ce.z_cf)

        return {
            "x_cf": x_cf,
            "z_cf": ce.z_cf,
            "score": ce.score,
            "meta": {
                "latent_distance": float(torch.linalg.norm(ce.z_cf - problem.z).item()),
                "constraint_penalty": float(ce.constraint_penalty),
                "constraint_metrics": ce.constraint_metrics,
                "robust": True,
                "optimizer": getattr(ce, "optimizer", None),
                "validity_target": target,
            },
        }

    return None


# -------------------------
# Pareto-specific helper
# -------------------------


def select_from_pareto(
    *,
    pareto: Sequence[Any],  # CandidateEval
    problem: Any,
    constraint_eval: Any,
    x_orig: Tensor,
    cfg: SelectionConfig,
    strategy: str = "knee",  # "knee" or "min_latent"
) -> Optional[Dict[str, Any]]:
    """
    Select from Pareto front using a simple, transparent rule.
    """
    if not pareto:
        return None

    if strategy == "min_latent":
        ordered = sorted(
            pareto,
            key=lambda ce: torch.linalg.norm(ce.z_cf - problem.z).item(),
        )
    elif strategy == "knee":
        # crude knee heuristic: minimize sum of normalized objectives
        objs = np.stack([ce.objectives for ce in pareto], axis=0)
        objs = (objs - objs.min(axis=0)) / (objs.ptp(axis=0) + 1e-9)
        scores = objs.sum(axis=1)
        ordered = [pareto[i] for i in np.argsort(scores)]
    else:
        raise ValueError("strategy must be 'knee' or 'min_latent'")

    return select_from_candidates(
        candidates=ordered,
        problem=problem,
        constraint_eval=constraint_eval,
        x_orig=x_orig,
        cfg=cfg,
    )


# -------------------------
# Minimal usage example
# -------------------------
if __name__ == "__main__":
    print("Part 5: Final Selection & Validation module loaded.")
    print("This selects robust, plausible CFs — not just threshold-crossers.")
