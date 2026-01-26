# objectives.py
# Part 3: Objective(s) Definition
#
# Responsibility:
# - Define SINGLE-objective and MULTI-objective formulations
# - Glue together:
#     • latent distance
#     • decoded validity (score vs tau)
#     • decoded constraint penalties (from Part 2)
#     • NormalCore proximity
#
# This module:
# - does NOT optimize
# - does NOT sample
# - does NOT decide feasibility
#
# External-first:
# - torch, numpy
# - sklearn (optional, for fast distance helpers)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from sklearn.metrics import pairwise_distances

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


Tensor = torch.Tensor


# -------------------------
# Distance helpers
# -------------------------


def latent_l2(z_cf: Tensor, z: Tensor) -> float:
    return float(torch.linalg.norm(z_cf - z).item())


def normalcore_distance(
    z_cf: Tensor,
    normalcore_z: Tensor,
    reduction: str = "min",
) -> float:
    """
    Distance from z_cf to NormalCore.
    reduction: 'min' (default) or 'mean'
    """
    if not _HAS_SKLEARN:
        # fallback torch implementation
        diffs = normalcore_z - z_cf.unsqueeze(0)
        dists = torch.linalg.norm(diffs, dim=1)
        if reduction == "min":
            return float(torch.min(dists).item())
        elif reduction == "mean":
            return float(torch.mean(dists).item())
        else:
            raise ValueError("reduction must be 'min' or 'mean'")

    z_np = z_cf.detach().cpu().numpy().reshape(1, -1)
    nc_np = normalcore_z.detach().cpu().numpy()
    d = pairwise_distances(z_np, nc_np, metric="euclidean")[0]
    return float(d.min() if reduction == "min" else d.mean())


# -------------------------
# Scalar (single-objective) loss
# -------------------------


@dataclass
class ScalarObjectiveConfig:
    alpha_latent: float = 1.0
    beta_validity: float = 10.0
    gamma_constraints: float = 1.0
    delta_normalcore: float = 1.0
    normalcore_reduction: str = "min"  # or "mean"

    def validate(self) -> None:
        for k, v in self.__dict__.items():
            if not isinstance(v, (float, int, str)):
                raise TypeError(f"{k} has invalid type")
        if self.normalcore_reduction not in {"min", "mean"}:
            raise ValueError("normalcore_reduction must be 'min' or 'mean'")


class ScalarLatentCFObjective:
    """
    Canonical scalar objective:
        L =
          α * ||z_cf - z||
        + β * max(0, score(x_cf) - (tau - eps))
        + γ * decoded_constraint_penalty
        + δ * dist(z_cf, NormalCore)
    """

    def __init__(
        self,
        *,
        config: ScalarObjectiveConfig,
    ):
        config.validate()
        self.cfg = config

    def __call__(
        self,
        *,
        z_cf: Tensor,
        z: Tensor,
        score_cf: float,
        validity_target: float,
        constraint_penalty: float,
        normalcore_z: Optional[Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Returns:
            total_loss, diagnostics
        """

        # Latent distance
        d_latent = latent_l2(z_cf, z)

        # Validity loss
        validity_excess = max(0.0, score_cf - validity_target)

        # NormalCore distance
        d_nc = 0.0
        if normalcore_z is not None:
            d_nc = normalcore_distance(
                z_cf, normalcore_z, reduction=self.cfg.normalcore_reduction
            )

        # Total loss
        loss = (
            self.cfg.alpha_latent * d_latent
            + self.cfg.beta_validity * validity_excess
            + self.cfg.gamma_constraints * constraint_penalty
            + self.cfg.delta_normalcore * d_nc
        )

        diagnostics = {
            "latent_distance": d_latent,
            "validity_excess": validity_excess,
            "constraint_penalty": constraint_penalty,
            "normalcore_distance": d_nc,
            "total_loss": loss,
        }

        return float(loss), diagnostics


# -------------------------
# Multi-objective formulation
# -------------------------


@dataclass
class MultiObjectiveValues:
    """
    Canonical objective vector for NSGA-II / Pareto search.
    All objectives are MINIMIZED.
    """

    latent_distance: float
    validity_excess: float
    constraint_penalty: float
    normalcore_distance: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.latent_distance,
                self.validity_excess,
                self.constraint_penalty,
                self.normalcore_distance,
            ],
            dtype=np.float64,
        )


class MultiObjectiveLatentCF:
    """
    Produces objective vectors instead of scalar loss.
    """

    def __init__(self, *, normalcore_reduction: str = "min"):
        if normalcore_reduction not in {"min", "mean"}:
            raise ValueError("normalcore_reduction must be 'min' or 'mean'")
        self.normalcore_reduction = normalcore_reduction

    def __call__(
        self,
        *,
        z_cf: Tensor,
        z: Tensor,
        score_cf: float,
        validity_target: float,
        constraint_penalty: float,
        normalcore_z: Optional[Tensor] = None,
    ) -> MultiObjectiveValues:
        latent_dist = latent_l2(z_cf, z)
        validity_excess = max(0.0, score_cf - validity_target)

        if normalcore_z is None:
            nc_dist = 0.0
        else:
            nc_dist = normalcore_distance(
                z_cf, normalcore_z, reduction=self.normalcore_reduction
            )

        return MultiObjectiveValues(
            latent_distance=latent_dist,
            validity_excess=validity_excess,
            constraint_penalty=constraint_penalty,
            normalcore_distance=nc_dist,
        )


# -------------------------
# Utility: epsilon-constraint check
# -------------------------


def is_valid_under_epsilon_constraint(
    objectives: MultiObjectiveValues,
    eps_validity: float = 0.0,
) -> bool:
    """
    Common rule for NSGA-II filtering:
    validity_excess <= eps_validity
    """
    return objectives.validity_excess <= eps_validity


# -------------------------
# Minimal usage example
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    D = 8

    z = torch.randn(D)
    z_cf = z + 0.1 * torch.randn(D)
    score_cf = 0.8
    validity_target = 1.0
    constraint_penalty = 0.2
    normalcore_z = torch.randn(32, D)

    # Scalar objective
    scalar_cfg = ScalarObjectiveConfig(
        alpha_latent=1.0,
        beta_validity=20.0,
        gamma_constraints=5.0,
        delta_normalcore=1.0,
    )
    scalar_obj = ScalarLatentCFObjective(config=scalar_cfg)
    loss, diag = scalar_obj(
        z_cf=z_cf,
        z=z,
        score_cf=score_cf,
        validity_target=validity_target,
        constraint_penalty=constraint_penalty,
        normalcore_z=normalcore_z,
    )
    print("scalar loss:", loss)
    print("diagnostics:", diag)

    # Multi-objective
    mo = MultiObjectiveLatentCF()
    obj_vals = mo(
        z_cf=z_cf,
        z=z,
        score_cf=score_cf,
        validity_target=validity_target,
        constraint_penalty=constraint_penalty,
        normalcore_z=normalcore_z,
    )
    print("multi-objective vector:", obj_vals.as_array())
