from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.signal import welch
from sklearn.decomposition import PCA


@dataclass
class HardConstraintResult:
    ok: bool
    reasons: Dict[str, Any]


@dataclass
class SoftConstraintMetrics:
    dyn_violation: float
    curvature_violation: float
    spectral_violation: float
    coupling_violation: float


class SensorConstraintManager:
    """
    Centralized sensor-physics and realism constraints.

    This class:
      - never modifies optimization logic
      - never decides counterfactual structure
      - only checks, measures, and optionally repairs feasibility
    """

    def __init__(
        self,
        *,
        normal_core: torch.Tensor,  # (K,L,F)
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

    # ------------------------------------------------------------------
    # Hard constraints
    # ------------------------------------------------------------------
    def hard_check(self, x: torch.Tensor, x_cf: torch.Tensor) -> HardConstraintResult:
        reasons: Dict[str, Any] = {}

        # immutable sensors
        if self.immutable_mask.any():
            if not torch.allclose(
                x_cf[:, self.immutable_mask],
                x[:, self.immutable_mask],
                atol=0.0,
                rtol=0.0,
            ):
                reasons["immutable_violation"] = True

        # value bounds
        if (x_cf < self.val_lo.unsqueeze(0)).any():
            reasons["below_min"] = True
        if (x_cf > self.val_hi.unsqueeze(0)).any():
            reasons["above_max"] = True

        ok = len(reasons) == 0
        return HardConstraintResult(ok=ok, reasons=reasons)

    # ------------------------------------------------------------------
    # Soft constraint metrics (objectives)
    # ------------------------------------------------------------------
    def soft_metrics(self, x_cf: torch.Tensor) -> SoftConstraintMetrics:
        # -------- Dynamic violation (1st derivative) --------
        dx = (x_cf[1:, :] - x_cf[:-1, :]).abs()
        dyn_v = torch.relu(dx - self.max_dx.unsqueeze(0)).mean().item()

        # -------- Curvature violation (2nd derivative) --------
        ddx = (dx[1:, :] - dx[:-1, :]).abs()
        curvature_v = ddx.mean().item()

        # -------- Spectral violation --------
        spec_v = 0.0
        for f in range(self.F):
            _, Pxx = welch(x_cf[:, f].detach().cpu().numpy())
            spec_v += max(0.0, Pxx.mean() - self.spectral_thr)
        spec_v /= float(self.F)

        # -------- Coupling violation (PCA residual) --------
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

    # ------------------------------------------------------------------
    # Optional repair / projection
    # ------------------------------------------------------------------
    def repair(self, x: torch.Tensor, x_cf: torch.Tensor) -> torch.Tensor:
        """
        Light-touch projection to keep candidates feasible.
        Does NOT attempt to optimize, only repairs obvious violations.
        """
        y = x_cf.clone()

        # clip values
        y = torch.max(torch.min(y, self.val_hi), self.val_lo)

        # enforce immutable sensors
        if self.immutable_mask.any():
            y[:, self.immutable_mask] = x[:, self.immutable_mask]

        return y
