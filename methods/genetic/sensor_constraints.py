from __future__ import annotations

# Postpones evaluation of type annotations.
# Useful for forward references and reduces runtime overhead.
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Used for spectral (frequency-domain) realism checks
from scipy.signal import welch

# Used to model inter-sensor coupling structure
from sklearn.decomposition import PCA


@dataclass
class HardConstraintResult:
    """
    Result of evaluating hard constraints.

    Fields:
    - ok: whether all hard constraints are satisfied
    - reasons: dictionary describing which constraints failed
    """

    ok: bool
    reasons: Dict[str, Any]


@dataclass
class SoftConstraintMetrics:
    """
    Quantitative soft constraint violations.

    These are NOT binary failures.
    They are continuous penalties used as optimization objectives.
    """

    dyn_violation: float  # Excessive first-derivative (rate-of-change)
    curvature_violation: float  # Excessive second-derivative (smoothness)
    spectral_violation: float  # Frequency-domain mismatch
    coupling_violation: float  # Cross-sensor coupling violation


class SensorConstraintManager:
    """
    Centralized sensor-physics and realism constraint module.

    Key design principles:
    - Completely independent of optimization strategy
    - Stateless with respect to genome or optimizer
    - Provides:
        * Hard feasibility checks
        * Soft realism metrics
        * Optional projection / repair
    - NEVER decides *what* to change
    - ONLY decides *whether* a signal is realistic
    """

    def __init__(
        self,
        *,
        normal_core: torch.Tensor,  # (K, L, F): reference normal signals
        value_quantiles: Tuple[float, float] = (0.01, 0.99),
        max_delta_quantile: float = 0.995,
        immutable_mask: Optional[torch.Tensor] = None,  # (F,)
        pca_rank: int = 8,
        coupling_quantile: float = 0.995,
        spectral_quantile: float = 0.995,
        device: str = "cpu",
    ):
        if normal_core.dim() != 3:
            raise ValueError("normal_core must have shape (K,L,F)")

        self.device = device
        self.normal_core = normal_core.to(device)
        self.K, self.L, self.F = self.normal_core.shape

        if immutable_mask is None:
            self.immutable_mask = torch.zeros(self.F, dtype=torch.bool, device=device)
        else:
            self.immutable_mask = immutable_mask.to(device).bool()

        # ---------------- Value bounds ----------------
        flat = self.normal_core.reshape(-1, self.F)
        qlo, qhi = value_quantiles
        self.val_lo = torch.quantile(flat, qlo, dim=0)
        self.val_hi = torch.quantile(flat, qhi, dim=0)

        # ---------------- Rate limits ----------------
        dx = (self.normal_core[:, 1:, :] - self.normal_core[:, :-1, :]).abs()
        dx = dx.reshape(-1, self.F)
        self.max_dx = torch.quantile(dx, max_delta_quantile, dim=0)

        # ---------------- PCA coupling ----------------
        X = self.normal_core.reshape(self.K, self.L * self.F).cpu().numpy()

        self.pca = PCA(
            n_components=min(pca_rank, X.shape[0], X.shape[1]), svd_solver="full"
        )

        Z = self.pca.fit_transform(X)
        X_rec = self.pca.inverse_transform(Z)
        resid = ((X - X_rec) ** 2).mean(axis=1)

        self.coupling_thr = float(np.quantile(resid, coupling_quantile))

        # ---------------- Spectral reference ----------------
        psd_vals = []

        for k in range(self.K):
            for f in range(self.F):
                _, Pxx = welch(self.normal_core[k, :, f].cpu().numpy())
                psd_vals.append(Pxx.mean())

        self.spectral_thr = float(np.quantile(psd_vals, spectral_quantile))

    def hard_check(self, x: torch.Tensor, x_cf: torch.Tensor) -> HardConstraintResult:
        reasons: Dict[str, Any] = {}
        if self.immutable_mask.any():
            if not torch.allclose(
                x_cf[:, self.immutable_mask],
                x[:, self.immutable_mask],
                atol=1e-6,
                rtol=0.0,
            ):
                reasons["immutable_violation"] = True

        if (x_cf < self.val_lo.unsqueeze(0)).any():
            reasons["below_min"] = True
        if (x_cf > self.val_hi.unsqueeze(0)).any():
            reasons["above_max"] = True

        ok = len(reasons) == 0
        return HardConstraintResult(ok=ok, reasons=reasons)

    def soft_metrics(self, x_cf: torch.Tensor) -> SoftConstraintMetrics:
        dx = (x_cf[1:, :] - x_cf[:-1, :]).abs()
        dyn_v = torch.relu(dx - self.max_dx.unsqueeze(0)).mean().item()
        ddx = (dx[1:, :] - dx[:-1, :]).abs()
        curvature_v = ddx.mean().item()
        spec_v = 0.0
        for f in range(self.F):
            _, Pxx = welch(x_cf[:, f].detach().cpu().numpy())
            spec_v += max(0.0, Pxx.mean() - self.spectral_thr)
        spec_v /= float(self.F)
        X = x_cf.reshape(1, self.L * self.F).cpu().numpy()
        Z = self.pca.transform(X)
        X_rec = self.pca.inverse_transform(Z)
        resid = float(((X - X_rec) ** 2).mean())
        coupling_v = max(0.0, resid - self.coupling_thr)
        return SoftConstraintMetrics(
            dyn_violation=float(dyn_v),
            curvature_violation=float(curvature_v),
            spectral_violation=float(spec_v),
            coupling_violation=float(coupling_v),
        )

    def repair(self, x: torch.Tensor, x_cf: torch.Tensor) -> torch.Tensor:
        """
        Lightweight feasibility repair.

        Purpose:
        - Keep optimizer from wasting effort
        - Enforce obvious constraints only
        - Does NOT attempt to improve objectives
        """
        y = x_cf.clone()
        y = torch.max(torch.min(y, self.val_hi), self.val_lo)
        if self.immutable_mask.any():
            y[:, self.immutable_mask] = x[:, self.immutable_mask]
        return y
