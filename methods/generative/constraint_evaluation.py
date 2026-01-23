# part3_constraint_evaluation.py
# Generative Infilling Counterfactuals — Part 3: Constraint Evaluation
#
# Responsibilities:
# - hard constraint checking (reject immediately)
# - soft constraint scoring (library-based realism metrics)
# - NO anomaly score logic here
#
# This module decides: plausible vs implausible — NOT valid CF vs invalid CF.
# Validity wrt threshold is handled in Part 4.
#
# Library-first:
# - numpy
# - scipy.signal
# - sklearn (PCA, scaling)
# - tslearn (DTW distances)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter, welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw


# ----------------------------
# Config
# ----------------------------
@dataclass
class ConstraintConfig:
    # hard constraints
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    max_abs_delta: Optional[float] = None  # max |x_cf - x| per cell
    immutable_features: Optional[Iterable[int]] = None

    # soft constraints
    max_rate_of_change: Optional[float] = None  # |dx/dt|
    curvature_weight: float = 1.0
    spectral_weight: float = 1.0
    coupling_weight: float = 1.0
    dtw_weight: float = 1.0

    # signal processing
    savgol_window: int = 7
    savgol_poly: int = 2
    psd_nperseg: int = 64

    # PCA coupling
    pca_components: int = 3


# ----------------------------
# Output
# ----------------------------
@dataclass
class ConstraintResult:
    feasible: bool
    hard_violations: Dict[str, float]
    soft_metrics: Dict[str, float]
    total_soft_penalty: float


# ----------------------------
# Evaluator
# ----------------------------
class ConstraintEvaluator:
    def __init__(
        self,
        cfg: ConstraintConfig,
        normalcore: Optional[np.ndarray] = None,  # (K,L,F)
    ):
        self.cfg = cfg
        self.immutable = (
            set(int(i) for i in cfg.immutable_features)
            if cfg.immutable_features
            else set()
        )

        # prepare PCA on normal core if provided
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        if normalcore is not None:
            self._fit_pca(normalcore)

    # ----------------------------
    # Public API
    # ----------------------------
    def evaluate(
        self,
        x_orig: np.ndarray,  # (L,F)
        x_cf: np.ndarray,  # (L,F)
        mask: np.ndarray,  # (L,F) bool
    ) -> ConstraintResult:
        x_orig = self._as_2d(x_orig)
        x_cf = self._as_2d(x_cf)
        mask = mask.astype(bool)

        hard = self._check_hard_constraints(x_orig, x_cf, mask)
        if hard:
            return ConstraintResult(
                feasible=False,
                hard_violations=hard,
                soft_metrics={},
                total_soft_penalty=float("inf"),
            )

        soft = self._compute_soft_constraints(x_orig, x_cf, mask)
        total_penalty = float(sum(soft.values()))

        return ConstraintResult(
            feasible=True,
            hard_violations={},
            soft_metrics=soft,
            total_soft_penalty=total_penalty,
        )

    # ----------------------------
    # Hard constraints
    # ----------------------------
    def _check_hard_constraints(
        self,
        x_orig: np.ndarray,
        x_cf: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, float]:
        violations: Dict[str, float] = {}

        # value bounds
        if self.cfg.value_min is not None:
            v = float(np.min(x_cf))
            if v < self.cfg.value_min:
                violations["value_min"] = v

        if self.cfg.value_max is not None:
            v = float(np.max(x_cf))
            if v > self.cfg.value_max:
                violations["value_max"] = v

        # immutable features unchanged
        for f in self.immutable:
            if f < 0 or f >= x_cf.shape[1]:
                continue
            delta = np.max(np.abs(x_cf[:, f] - x_orig[:, f]))
            if delta > 0:
                violations[f"immutable_feature_{f}"] = float(delta)

        # max absolute delta
        if self.cfg.max_abs_delta is not None:
            d = np.max(np.abs(x_cf - x_orig))
            if d > self.cfg.max_abs_delta:
                violations["max_abs_delta"] = float(d)

        return violations

    # ----------------------------
    # Soft constraints
    # ----------------------------
    def _compute_soft_constraints(
        self,
        x_orig: np.ndarray,
        x_cf: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # consider only masked region for most metrics
        if not np.any(mask):
            return metrics

        # rate-of-change
        if self.cfg.max_rate_of_change is not None:
            dx = np.diff(x_cf, axis=0)
            roc = np.max(np.abs(dx))
            metrics["rate_of_change"] = max(
                0.0,
                roc - float(self.cfg.max_rate_of_change),
            )

        # curvature (second derivative roughness)
        curv = self._curvature_penalty(x_cf, mask)
        metrics["curvature"] = self.cfg.curvature_weight * curv

        # spectral realism
        spec = self._spectral_penalty(x_orig, x_cf, mask)
        metrics["spectral"] = self.cfg.spectral_weight * spec

        # cross-sensor coupling
        if self.pca is not None:
            coup = self._coupling_penalty(x_cf)
            metrics["coupling"] = self.cfg.coupling_weight * coup

        # DTW distance to original (masked region only)
        dtw_p = self._dtw_penalty(x_orig, x_cf, mask)
        metrics["dtw"] = self.cfg.dtw_weight * dtw_p

        return metrics

    # ----------------------------
    # Individual penalties
    # ----------------------------
    def _curvature_penalty(self, x: np.ndarray, mask: np.ndarray) -> float:
        L, F = x.shape
        penalty = 0.0
        for f in range(F):
            y = x[:, f]
            if L >= self.cfg.savgol_window:
                y_s = savgol_filter(
                    y,
                    window_length=self.cfg.savgol_window,
                    polyorder=self.cfg.savgol_poly,
                    mode="interp",
                )
                curv = np.diff(y_s, n=2)
                penalty += float(np.mean(np.abs(curv)))
        return penalty / float(F)

    def _spectral_penalty(
        self,
        x_orig: np.ndarray,
        x_cf: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        L, F = x_cf.shape
        p = 0.0
        for f in range(F):
            _, pxx_o = welch(x_orig[:, f], nperseg=min(self.cfg.psd_nperseg, L))
            _, pxx_c = welch(x_cf[:, f], nperseg=min(self.cfg.psd_nperseg, L))
            # normalize
            pxx_o = pxx_o / (np.sum(pxx_o) + 1e-8)
            pxx_c = pxx_c / (np.sum(pxx_c) + 1e-8)
            p += float(np.mean(np.abs(pxx_o - pxx_c)))
        return p / float(F)

    def _coupling_penalty(self, x_cf: np.ndarray) -> float:
        X = self.scaler.transform(x_cf)
        Xp = self.pca.transform(X)
        Xr = self.pca.inverse_transform(Xp)
        resid = np.mean((X - Xr) ** 2)
        return float(resid)

    def _dtw_penalty(
        self,
        x_orig: np.ndarray,
        x_cf: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        # restrict to masked timesteps
        idx = np.any(mask, axis=1)
        if not np.any(idx):
            return 0.0
        d = 0.0
        F = x_cf.shape[1]
        for f in range(F):
            d += dtw(x_orig[idx, f], x_cf[idx, f])
        return float(d / F)

    # ----------------------------
    # PCA fitting
    # ----------------------------
    def _fit_pca(self, normalcore: np.ndarray) -> None:
        K, L, F = normalcore.shape
        X = normalcore.reshape(K * L, F)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.pca = PCA(
            n_components=min(self.cfg.pca_components, F),
            svd_solver="full",
        ).fit(Xs)

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected (L,F), got {x.shape}")
        return x


# ----------------------------
# Minimal self-test
# ----------------------------
if __name__ == "__main__":
    L, F = 80, 3
    x = np.sin(np.linspace(0, 4 * np.pi, L))[:, None] * np.ones((1, F))
    x_cf = x.copy()
    x_cf[30:40] += 0.5

    mask = np.zeros((L, F), dtype=bool)
    mask[30:40, :] = True

    normalcore = np.stack([x + 0.01 * np.random.randn(L, F) for _ in range(10)], axis=0)

    cfg = ConstraintConfig(
        value_min=-2.0,
        value_max=2.0,
        max_abs_delta=1.0,
        max_rate_of_change=0.5,
        immutable_features=[2],
    )
    ev = ConstraintEvaluator(cfg, normalcore=normalcore)
    res = ev.evaluate(x, x_cf, mask)

    print("Feasible:", res.feasible)
    print("Hard:", res.hard_violations)
    print("Soft:", res.soft_metrics)
    print("Total penalty:", res.total_soft_penalty)
