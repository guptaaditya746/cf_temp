# decoded_constraints.py
# Part 2: Constraint Evaluation (Decoded Space)
#
# Responsibility:
# - Evaluate decoded candidates x_cf = Decoder(z_cf)
# - Enforce HARD constraints (binary reject)
# - Compute SOFT constraint penalties + diagnostics
#
# This module is intentionally model-agnostic.
# It knows NOTHING about optimization or objectives.
#
# External libraries FIRST:
# - numpy
# - torch
# - scipy (signal / fft utilities)
# - tslearn (optional, guarded import)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.signal import welch

try:
    from tslearn.metrics import dtw

    _HAS_TSLEARN = True
except Exception:
    _HAS_TSLEARN = False


Tensor = torch.Tensor


# -------------------------
# Helper checks
# -------------------------


def _check_finite(x: Tensor) -> bool:
    return torch.isfinite(x).all().item()


def _l2_smoothness(x: Tensor) -> float:
    """
    Second-order temporal smoothness penalty.
    """
    # x: (L,F)
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    return float(torch.mean(d2**2).item())


def _rate_of_change_violation(x: Tensor, max_delta: Tensor) -> float:
    """
    Penalize rate-of-change violations.
    max_delta: (F,) tensor with allowed per-feature delta per timestep
    """
    dx = torch.abs(x[1:] - x[:-1])  # (L-1,F)
    viol = torch.relu(dx - max_delta)
    return float(torch.mean(viol).item())


def _spectral_deviation(
    x: Tensor,
    ref_psd: np.ndarray,
    fs: float = 1.0,
) -> float:
    """
    Compare PSD of x to reference PSD using L2 distance.
    """
    x_np = x.detach().cpu().numpy()
    psd_cf = []
    for f in range(x_np.shape[1]):
        _, p = welch(x_np[:, f], fs=fs, nperseg=min(256, x_np.shape[0]))
        psd_cf.append(p)
    psd_cf = np.stack(psd_cf, axis=0)
    return float(np.mean((psd_cf - ref_psd) ** 2))


def _coupling_violation(x: Tensor, coupling_matrix: Tensor) -> float:
    """
    Penalize violation of linear cross-sensor coupling.
    coupling_matrix: (F,F), expected correlation / coupling strength
    """
    # empirical correlation
    x_np = x.detach().cpu().numpy()
    corr = np.corrcoef(x_np.T)
    diff = corr - coupling_matrix.detach().cpu().numpy()
    return float(np.mean(diff**2))


# -------------------------
# Constraint specification
# -------------------------


@dataclass(frozen=True)
class DecodedConstraintSpec:
    """
    Declarative constraint specification.
    All tensors refer to decoded space (L,F) or feature-level (F,).
    """

    # HARD constraints
    value_min: Optional[Tensor] = None  # (F,)
    value_max: Optional[Tensor] = None  # (F,)
    immutable_mask: Optional[Tensor] = None  # (L,F) True = cannot change

    # SOFT constraints
    max_rate_of_change: Optional[Tensor] = None  # (F,)
    smoothness_weight: float = 1.0
    roc_weight: float = 1.0
    spectral_weight: float = 0.0
    coupling_weight: float = 0.0
    dtw_weight: float = 0.0

    # Reference data for soft constraints
    spectral_ref_psd: Optional[np.ndarray] = None  # (F, P)
    coupling_matrix: Optional[Tensor] = None  # (F,F)
    reference_window: Optional[Tensor] = None  # (L,F), for DTW

    def validate(self, L: int, F: int) -> None:
        if self.value_min is not None and self.value_min.shape != (F,):
            raise ValueError("value_min must have shape (F,)")
        if self.value_max is not None and self.value_max.shape != (F,):
            raise ValueError("value_max must have shape (F,)")
        if self.immutable_mask is not None and self.immutable_mask.shape != (L, F):
            raise ValueError("immutable_mask must have shape (L,F)")
        if self.max_rate_of_change is not None and self.max_rate_of_change.shape != (
            F,
        ):
            raise ValueError("max_rate_of_change must have shape (F,)")
        if self.dtw_weight > 0 and not _HAS_TSLEARN:
            raise RuntimeError("dtw_weight > 0 requires tslearn to be installed")


# -------------------------
# Constraint evaluator
# -------------------------


@dataclass
class ConstraintEvaluationResult:
    hard_ok: bool
    soft_penalty: float
    metrics: Dict[str, float]
    failure_reason: Optional[str] = None


class DecodedConstraintEvaluator:
    """
    Evaluates decoded candidates against hard + soft constraints.
    """

    def __init__(self, spec: DecodedConstraintSpec):
        self.spec = spec

    def evaluate(
        self,
        x_cf: Tensor,
        x_orig: Optional[Tensor] = None,
    ) -> ConstraintEvaluationResult:
        """
        x_cf: decoded candidate (L,F)
        x_orig: original window (L,F), required if immutable_mask or DTW used
        """
        if x_cf.ndim != 2:
            return ConstraintEvaluationResult(
                hard_ok=False,
                soft_penalty=float("inf"),
                metrics={},
                failure_reason="invalid_shape",
            )

        L, F = x_cf.shape
        self.spec.validate(L, F)

        # -------------------------
        # HARD constraints
        # -------------------------
        if not _check_finite(x_cf):
            return ConstraintEvaluationResult(
                hard_ok=False,
                soft_penalty=float("inf"),
                metrics={},
                failure_reason="non_finite_values",
            )

        if self.spec.value_min is not None:
            if (x_cf < self.spec.value_min).any():
                return ConstraintEvaluationResult(
                    hard_ok=False,
                    soft_penalty=float("inf"),
                    metrics={},
                    failure_reason="value_min_violation",
                )

        if self.spec.value_max is not None:
            if (x_cf > self.spec.value_max).any():
                return ConstraintEvaluationResult(
                    hard_ok=False,
                    soft_penalty=float("inf"),
                    metrics={},
                    failure_reason="value_max_violation",
                )

        if self.spec.immutable_mask is not None:
            if x_orig is None:
                raise ValueError("x_orig required when immutable_mask is set")
            diff = torch.abs(x_cf - x_orig)
            if (diff[self.spec.immutable_mask] > 1e-6).any():
                return ConstraintEvaluationResult(
                    hard_ok=False,
                    soft_penalty=float("inf"),
                    metrics={},
                    failure_reason="immutable_violation",
                )

        # -------------------------
        # SOFT constraints
        # -------------------------
        metrics: Dict[str, float] = {}
        penalty = 0.0

        # Smoothness
        sm = _l2_smoothness(x_cf)
        metrics["smoothness"] = sm
        penalty += self.spec.smoothness_weight * sm

        # Rate of change
        if self.spec.max_rate_of_change is not None:
            roc = _rate_of_change_violation(x_cf, self.spec.max_rate_of_change)
            metrics["rate_of_change"] = roc
            penalty += self.spec.roc_weight * roc

        # Spectral realism
        if self.spec.spectral_weight > 0 and self.spec.spectral_ref_psd is not None:
            sp = _spectral_deviation(x_cf, self.spec.spectral_ref_psd)
            metrics["spectral_dev"] = sp
            penalty += self.spec.spectral_weight * sp

        # Cross-sensor coupling
        if self.spec.coupling_weight > 0 and self.spec.coupling_matrix is not None:
            cp = _coupling_violation(x_cf, self.spec.coupling_matrix)
            metrics["coupling_dev"] = cp
            penalty += self.spec.coupling_weight * cp

        # DTW similarity
        if self.spec.dtw_weight > 0 and self.spec.reference_window is not None:
            if x_orig is None:
                raise ValueError("x_orig required for DTW penalty")
            dtw_dist = 0.0
            x_cf_np = x_cf.detach().cpu().numpy()
            x_ref_np = self.spec.reference_window.detach().cpu().numpy()
            for f in range(F):
                dtw_dist += dtw(x_cf_np[:, f], x_ref_np[:, f])
            dtw_dist /= F
            metrics["dtw"] = float(dtw_dist)
            penalty += self.spec.dtw_weight * dtw_dist

        return ConstraintEvaluationResult(
            hard_ok=True,
            soft_penalty=float(penalty),
            metrics=metrics,
            failure_reason=None,
        )


# -------------------------
# Minimal usage example
# -------------------------
if __name__ == "__main__":
    L, F = 32, 3
    torch.manual_seed(0)

    x_orig = torch.randn(L, F)
    x_cf = x_orig.clone()
    x_cf[10:15, 0] += 0.5  # small perturbation

    spec = DecodedConstraintSpec(
        value_min=torch.tensor([-5.0, -5.0, -5.0]),
        value_max=torch.tensor([5.0, 5.0, 5.0]),
        max_rate_of_change=torch.tensor([1.0, 1.0, 1.0]),
        smoothness_weight=1.0,
        roc_weight=1.0,
    )

    evaluator = DecodedConstraintEvaluator(spec)
    res = evaluator.evaluate(x_cf, x_orig=x_orig)

    print("hard_ok:", res.hard_ok)
    print("soft_penalty:", res.soft_penalty)
    print("metrics:", res.metrics)
