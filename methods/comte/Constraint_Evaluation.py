# Part 4 — Constraint Evaluation
# - hard feasibility checks (reject immediately)
# - soft realism metrics (used for ranking / optimization)
#
# HARD RULES:
# - NO silent acceptance
# - constraints evaluated in DECODED space
# - model score is NOT a realism constraint here (handled in Part 5)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Tensor = torch.Tensor


# -------------------------------
# Config & containers
# -------------------------------


@dataclass
class ConstraintConfig:
    # hard constraints
    sensor_min: Optional[float] = None
    sensor_max: Optional[float] = None
    immutable_sensors: Optional[List[int]] = None

    # soft constraints
    max_delta: Optional[float] = None  # rate-of-change limit
    pca_components: int = 5  # cross-sensor coupling
    pca_ref_samples: int = 512  # how many NormalCore points to fit PCA
    boundary_window: int = 2  # timesteps near boundaries


@dataclass
class ConstraintResult:
    feasible: bool
    hard_violations: Dict[str, float]
    soft_metrics: Dict[str, float]


# -------------------------------
# Utilities
# -------------------------------


def _rate_of_change_violation(x: Tensor) -> float:
    # mean absolute first difference
    dx = torch.diff(x, dim=0)
    return float(dx.abs().mean())


def _boundary_discontinuity(
    x_orig: Tensor,
    x_cf: Tensor,
    seg: Tuple[int, int],
    w: int,
) -> float:
    s, t = seg
    vals = []

    for i in range(1, w + 1):
        if s - i >= 0:
            vals.append(torch.norm(x_cf[s] - x_orig[s - i], p=2).item())
        if t - 1 + i < x_cf.shape[0]:
            vals.append(torch.norm(x_cf[t - 1] - x_orig[t - 1 + i], p=2).item())

    return float(np.mean(vals)) if vals else 0.0


# -------------------------------
# Core evaluator
# -------------------------------


class ConstraintEvaluator:
    def __init__(
        self,
        normal_core: Tensor,  # (K,L,F)
        cfg: Optional[ConstraintConfig] = None,
    ):
        if normal_core.ndim != 3:
            raise ValueError("NormalCore must be (K,L,F)")
        self.normal_core = normal_core.detach().float().cpu()
        self.cfg = cfg or ConstraintConfig()

        # fit PCA once (flattened time)
        self._fit_pca()

    def _fit_pca(self):
        cfg = self.cfg
        K, L, F = self.normal_core.shape

        X = self.normal_core.reshape(-1, F).numpy()
        if cfg.pca_ref_samples and X.shape[0] > cfg.pca_ref_samples:
            idx = np.random.choice(X.shape[0], cfg.pca_ref_samples, replace=False)
            X = X[idx]

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.pca = PCA(
            n_components=min(cfg.pca_components, Xs.shape[1]),
            random_state=0,
        ).fit(Xs)

    # ---------------------------

    def evaluate(
        self,
        x_orig: Tensor,  # (L,F)
        x_cf: Tensor,  # (L,F)
        replaced_segment: Tuple[int, int],
    ) -> ConstraintResult:
        cfg = self.cfg

        hard_violations: Dict[str, float] = {}
        soft_metrics: Dict[str, float] = {}

        # ---------- HARD CONSTRAINTS ----------

        # bounds
        if cfg.sensor_min is not None:
            v = float((x_cf < cfg.sensor_min).sum().item())
            if v > 0:
                hard_violations["sensor_min"] = v

        if cfg.sensor_max is not None:
            v = float((x_cf > cfg.sensor_max).sum().item())
            if v > 0:
                hard_violations["sensor_max"] = v

        # immutable sensors unchanged
        if cfg.immutable_sensors:
            diff = x_cf[:, cfg.immutable_sensors] - x_orig[:, cfg.immutable_sensors]
            v = float(diff.abs().sum().item())
            if v > 0:
                hard_violations["immutable_sensors"] = v

        feasible = len(hard_violations) == 0

        # ---------- SOFT CONSTRAINTS ----------

        # rate-of-change
        roc = _rate_of_change_violation(x_cf)
        soft_metrics["rate_of_change"] = roc
        if cfg.max_delta is not None:
            soft_metrics["rate_excess"] = max(0.0, roc - cfg.max_delta)

        # boundary smoothness
        bd = _boundary_discontinuity(
            x_orig=x_orig,
            x_cf=x_cf,
            seg=replaced_segment,
            w=cfg.boundary_window,
        )
        soft_metrics["boundary_discontinuity"] = bd

        # cross-sensor coupling (PCA reconstruction error)
        X = x_cf.detach().cpu().numpy()
        Xs = self.scaler.transform(X)
        Xp = self.pca.inverse_transform(self.pca.transform(Xs))
        pca_err = float(np.mean((Xs - Xp) ** 2))
        soft_metrics["pca_reconstruction_error"] = pca_err

        return ConstraintResult(
            feasible=feasible,
            hard_violations=hard_violations,
            soft_metrics=soft_metrics,
        )


# -------------------------------
# Example minimal usage
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    K, L, F = 10, 128, 6
    normal_core = torch.randn(K, L, F)

    x = torch.randn(L, F)
    x_cf = x.clone()
    x_cf[40:56] += 0.5  # fake substitution

    evaluator = ConstraintEvaluator(
        normal_core,
        ConstraintConfig(
            sensor_min=-5.0,
            sensor_max=5.0,
            immutable_sensors=[0],
            max_delta=2.0,
        ),
    )

    res = evaluator.evaluate(
        x_orig=x,
        x_cf=x_cf,
        replaced_segment=(40, 56),
    )

    assert isinstance(res.feasible, bool)
    assert isinstance(res.soft_metrics, dict)
