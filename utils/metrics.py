# cf_metrics.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# -----------------------------
# Shape helpers
# -----------------------------
def _to_2d(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    if x.dim() == 2:
        return x
    if x.dim() == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError(f"Expected (L,F) or (1,L,F), got {tuple(x.shape)}")


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x.unsqueeze(0)
    raise ValueError(f"Expected (K,L,F) or (L,F), got {tuple(x.shape)}")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return float(x.detach().reshape(-1)[0].cpu().item())
    try:
        return float(x)
    except Exception:
        return None


# -----------------------------
# Basic distances
# -----------------------------
def _rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x - y) ** 2) + 1e-12)


def _mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def _max_abs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(x - y))


def _trimmed_mean_abs(delta: torch.Tensor, trim_q: float = 0.99) -> torch.Tensor:
    """
    Robust mean(|delta|) ignoring top (1-trim_q) fraction of absolute changes.
    delta: (L,F)
    """
    flat = torch.abs(delta).reshape(-1)
    if flat.numel() == 0:
        return torch.tensor(0.0, device=delta.device)
    thr = torch.quantile(flat, trim_q)
    kept = flat[flat <= thr]
    if kept.numel() == 0:
        return thr  # fallback
    return kept.mean()


def _trimmed_rmse(delta: torch.Tensor, trim_q: float = 0.99) -> torch.Tensor:
    """
    Robust RMSE ignoring top (1-trim_q) fraction of squared errors.
    delta: (L,F) = x_cf - x
    """
    flat = (delta**2).reshape(-1)
    if flat.numel() == 0:
        return torch.tensor(0.0, device=delta.device)
    thr = torch.quantile(flat, trim_q)
    kept = flat[flat <= thr]
    if kept.numel() == 0:
        return torch.sqrt(thr + 1e-12)
    return torch.sqrt(kept.mean() + 1e-12)


# -----------------------------
# Derivatives / smoothness
# -----------------------------
def _diff1(x: torch.Tensor) -> torch.Tensor:
    return x[1:] - x[:-1]


def _diff2(x: torch.Tensor) -> torch.Tensor:
    d1 = _diff1(x)
    return d1[1:] - d1[:-1]


def _tv_l1_first_diff(d: torch.Tensor) -> torch.Tensor:
    """
    Total variation (L1) on first differences of edit signal.
    d: (L,F)
    """
    if d.shape[0] < 2:
        return torch.tensor(0.0, device=d.device)
    d1 = _diff1(d)
    return torch.mean(torch.abs(d1))


def _boundary_discontinuity(d: torch.Tensor, changed_t: torch.Tensor) -> float:
    """
    Measures jump at boundaries of edited region(s): large jumps are less plausible.
    d: edit signal (L,F)
    changed_t: (L,) bool
    """
    idx = torch.nonzero(changed_t, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return 0.0
    idx_sorted = torch.sort(idx).values
    # boundary points: start and end neighbors of each segment
    diffs = idx_sorted[1:] - idx_sorted[:-1]
    breaks = torch.nonzero(diffs > 1, as_tuple=False).reshape(-1)
    seg_starts = torch.cat([idx_sorted[:1], idx_sorted[breaks + 1]])
    seg_ends = torch.cat([idx_sorted[breaks], idx_sorted[-1:]])  # inclusive

    vals = []
    L = d.shape[0]
    for s, e in zip(seg_starts.tolist(), seg_ends.tolist()):
        # discontinuity at left boundary: d[s] - d[s-1]
        if s - 1 >= 0:
            vals.append(torch.mean(torch.abs(d[s] - d[s - 1])))
        # discontinuity at right boundary: d[e+1] - d[e]
        if e + 1 < L:
            vals.append(torch.mean(torch.abs(d[e + 1] - d[e])))

    if not vals:
        return 0.0
    return float(torch.stack(vals).mean().detach().cpu().item())


# -----------------------------
# Segment stats
# -----------------------------
def _segment_stats_from_mask(mask: torch.Tensor) -> Dict[str, Any]:
    # mask: (L,) bool
    L = int(mask.numel())
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return {
            "changed_any": False,
            "n_changed_time": 0,
            "frac_changed_time": 0.0,
            "n_segments_time": 0,
            "max_segment_len_time": 0,
            "min_segment_len_time": 0,
            "mean_segment_len_time": 0.0,
            "segment_lengths_time": [],
            "first_change_t": None,
            "last_change_t": None,
            "contiguous_time": True,
        }

    idx_sorted = torch.sort(idx).values
    diffs = idx_sorted[1:] - idx_sorted[:-1]
    breaks = torch.nonzero(diffs > 1, as_tuple=False).reshape(-1)

    seg_starts = torch.cat([idx_sorted[:1], idx_sorted[breaks + 1]])
    seg_ends = torch.cat([idx_sorted[breaks], idx_sorted[-1:]])  # inclusive
    seg_lens = (seg_ends - seg_starts + 1).to(torch.long)

    n_segs = int(seg_lens.numel())
    seg_lens_list = [int(v.item()) for v in seg_lens]
    max_len = int(seg_lens.max().item())
    min_len = int(seg_lens.min().item())
    mean_len = float(seg_lens.float().mean().item())

    return {
        "changed_any": True,
        "n_changed_time": int(idx_sorted.numel()),
        "frac_changed_time": float(idx_sorted.numel() / max(1, L)),
        "n_segments_time": n_segs,
        "max_segment_len_time": max_len,
        "min_segment_len_time": min_len,
        "mean_segment_len_time": mean_len,
        "segment_lengths_time": seg_lens_list,
        "first_change_t": int(idx_sorted[0].item()),
        "last_change_t": int(idx_sorted[-1].item()),
        "contiguous_time": (n_segs == 1),
    }


def _span_from_mask(mask: torch.Tensor) -> Optional[Tuple[int, int]]:
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return None
    idx_sorted = torch.sort(idx).values
    return int(idx_sorted[0].item()), int(idx_sorted[-1].item())


# -----------------------------
# Normal-core plausibility helpers
# -----------------------------
def _feature_bounds_from_normal_core(
    normal_core: torch.Tensor,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nc = _to_3d(normal_core)
    flat = nc.reshape(-1, nc.shape[-1])  # (K*L, F)
    lo = torch.quantile(flat, q_lo, dim=0)
    hi = torch.quantile(flat, q_hi, dim=0)
    eps = 1e-6
    hi = torch.maximum(hi, lo + eps)
    return lo, hi


def _nn_plausibility_dist(
    x_cf: torch.Tensor,
    normal_core: torch.Tensor,
    metric: str = "rmse",
    chunk: int = 256,
    time_slice: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Distance from x_cf to nearest window in NormalCore (K,L,F).
    If time_slice is provided, compares only that span [t0,t1] (inclusive).
    """
    x2 = _to_2d(x_cf)
    nc = _to_3d(normal_core)
    if time_slice is not None:
        t0, t1 = time_slice
        x2 = x2[t0 : t1 + 1]
        nc = nc[:, t0 : t1 + 1, :]

    K = nc.shape[0]
    best = None
    for i in range(0, K, chunk):
        batch = nc[i : i + chunk]  # (B,l,F)
        if metric == "mae":
            d = torch.mean(torch.abs(batch - x2.unsqueeze(0)), dim=(1, 2))
        else:
            d = torch.sqrt(
                torch.mean((batch - x2.unsqueeze(0)) ** 2, dim=(1, 2)) + 1e-12
            )
        m = d.min()
        best = m if best is None else torch.minimum(best, m)

    return float(best.detach().cpu().item()) if best is not None else float("inf")


def _zscore_plausibility(
    x_cf: torch.Tensor,
    normal_core: torch.Tensor,
    eps: float = 1e-6,
    time_slice: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Mean/max abs z-score using NormalCore flattened across time and windows.
    If time_slice is provided, compute on that span only.
    """
    x2 = _to_2d(x_cf)
    nc = _to_3d(normal_core)
    flat = nc.reshape(-1, nc.shape[-1])  # (K*L, F)
    mu = flat.mean(dim=0)
    sd = flat.std(dim=0).clamp_min(eps)

    if time_slice is not None:
        t0, t1 = time_slice
        x2 = x2[t0 : t1 + 1]

    z = (x2 - mu) / sd
    absz = torch.abs(z)
    return {
        "z_abs_mean": float(absz.mean().detach().cpu().item()),
        "z_abs_max": float(absz.max().detach().cpu().item()),
    }


def _robust_z_mad(
    x_cf: torch.Tensor,
    normal_core: torch.Tensor,
    eps: float = 1e-6,
    time_slice: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Robust z using median and MAD (scaled by 1.4826).
    This is less sensitive to heavy tails / outliers than mean/std.
    """
    x2 = _to_2d(x_cf)
    nc = _to_3d(normal_core)
    flat = nc.reshape(-1, nc.shape[-1])  # (K*L, F)
    med = flat.median(dim=0).values
    mad = (flat - med).abs().median(dim=0).values
    sd = (1.4826 * mad).clamp_min(eps)

    if time_slice is not None:
        t0, t1 = time_slice
        x2 = x2[t0 : t1 + 1]

    z = (x2 - med) / sd
    absz = z.abs()
    return {
        "robust_z_abs_mean": float(absz.mean().detach().cpu().item()),
        "robust_z_abs_max": float(absz.max().detach().cpu().item()),
    }


def _mahalanobis_plausibility(
    x_cf: torch.Tensor,
    normal_core: torch.Tensor,
    eps: float = 1e-6,
    time_slice: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Correlation-aware plausibility: per-time Mahalanobis distance in feature space.
    Fit mean/cov from NormalCore flattened across windows+time.
    Then compute mean/max Mahalanobis over time points (or time_slice).
    """
    x2 = _to_2d(x_cf)
    nc = _to_3d(normal_core)
    flat = nc.reshape(-1, nc.shape[-1]).double()  # (N,F)
    mu = flat.mean(dim=0)

    # Covariance (F,F) - small (e.g., F=6), safe to invert with regularization.
    xc = flat - mu
    cov = (xc.T @ xc) / max(1, flat.shape[0] - 1)
    cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    inv = torch.linalg.pinv(cov)

    if time_slice is not None:
        t0, t1 = time_slice
        x2 = x2[t0 : t1 + 1]

    x2d = x2.double()
    y = x2d - mu
    # per-time distance: sqrt(y_t^T inv y_t)
    d2 = torch.einsum("tf,fg,tg->t", y, inv, y).clamp_min(0.0)
    d = torch.sqrt(d2 + 1e-12)
    return {
        "mahal_mean": float(d.mean().detach().cpu().item()),
        "mahal_max": float(d.max().detach().cpu().item()),
    }


# -----------------------------
# Scaler handling (scaler.json)
# -----------------------------
@dataclass(frozen=True)
class ScalerSpec:
    mean: torch.Tensor  # (F,)
    std: torch.Tensor  # (F,)
    features: Optional[List[str]] = None

    def to(self, device: torch.device) -> "ScalerSpec":
        return ScalerSpec(
            mean=self.mean.to(device), std=self.std.to(device), features=self.features
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std.clamp_min(1e-12)


def load_scaler_json(path: str, device: Optional[torch.device] = None) -> ScalerSpec:
    with open(path, "r") as f:
        obj = json.load(f)

    mean = obj.get("mean", obj.get("mu", obj.get("center")))
    std = obj.get("std", obj.get("sigma", obj.get("scale")))
    feats = obj.get("features", None)

    if mean is None or std is None:
        raise ValueError(
            f"Scaler JSON missing mean/std keys. Found keys: {list(obj.keys())}"
        )

    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32).clamp_min(1e-12)
    if device is not None:
        mean_t = mean_t.to(device)
        std_t = std_t.to(device)

    return ScalerSpec(mean=mean_t, std=std_t, features=feats)


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class MetricsConfig:
    eps_change: float = 1e-6

    # Bounds plausibility from normal core
    bounds_q_lo: float = 0.01
    bounds_q_hi: float = 0.99

    # NN plausibility
    nn_metric: str = "rmse"  # "rmse" or "mae"
    nn_chunk: int = 256

    # Robust proximity
    trim_q: float = 0.99

    # Threshold robustness checks
    thr_robust_fracs: Tuple[float, ...] = (0.01, 0.03, 0.05)  # +/-1%,3%,5%


class CounterfactualMetrics:
    def __init__(self, config: MetricsConfig = MetricsConfig()):
        self.cfg = config

    def compute(
        self,
        x: torch.Tensor,
        cf_result: Optional[Dict[str, Any]],
        threshold: float,
        normal_core: Optional[torch.Tensor] = None,
        scaler: Optional[Union[ScalerSpec, Dict[str, Any]]] = None,
        scaler_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        topk_features: int = 3,
    ) -> Dict[str, Any]:
        """
        x: (L,F) or (1,L,F)
        cf_result: {"x_cf": Tensor(L,F), "score": float, "meta": dict} or None
        threshold: scalar
        normal_core: (K,L,F) optional, used for plausibility metrics
        scaler/scaler_path: enables scaled-space metrics (recommended).
        feature_names: optional for reporting top changed features.
        """
        x0 = _to_2d(x).detach()
        thr = float(threshold)

        out: Dict[str, Any] = {
            "found": cf_result is not None,
            "threshold": thr,
        }

        # Build scaler spec if provided
        scaler_spec: Optional[ScalerSpec] = None
        if scaler_path is not None:
            scaler_spec = load_scaler_json(scaler_path, device=x0.device)
        elif isinstance(scaler, ScalerSpec):
            scaler_spec = scaler.to(x0.device)
        elif isinstance(scaler, dict):
            # allow passing {"mean":[...], "std":[...]} etc.
            tmp_path = None
            # try to interpret dict directly
            mean = scaler.get("mean", scaler.get("mu", scaler.get("center", None)))
            std = scaler.get("std", scaler.get("sigma", scaler.get("scale", None)))
            if mean is not None and std is not None:
                scaler_spec = ScalerSpec(
                    mean=torch.tensor(mean, dtype=torch.float32, device=x0.device),
                    std=torch.tensor(
                        std, dtype=torch.float32, device=x0.device
                    ).clamp_min(1e-12),
                )

        if cf_result is None:
            out.update(
                {
                    "valid": False,
                    "score_cf": None,
                    "delta_score_to_thr": None,
                    "margin_to_thr": None,
                    "valid_robust": None,
                    "dist_rmse": None,
                    "dist_mae": None,
                    "dist_max_abs": None,
                    "dist_trimmed_rmse": None,
                    "dist_trimmed_mae": None,
                    "dist_rmse_scaled": None,
                    "dist_mae_scaled": None,
                    "dist_trimmed_rmse_scaled": None,
                    "dist_trimmed_mae_scaled": None,
                    "per_feature_rmse": None,
                    "per_feature_mae": None,
                    "top_changed_features": None,
                    "n_changed_time": None,
                    "frac_changed_time": None,
                    "n_segments_time": None,
                    "max_segment_len_time": None,
                    "contiguous_time": None,
                    "n_changed_feat": None,
                    "frac_changed_feat": None,
                    "n_changed_cell": None,
                    "frac_changed_cell": None,
                    "smooth_l2_d1": None,
                    "smooth_l2_d2": None,
                    "tv_l1_d1": None,
                    "boundary_discontinuity": None,
                    "bounds_violations": None,
                    "bounds_violation_frac": None,
                    "bounds_violations_edited": None,
                    "bounds_violation_frac_edited": None,
                    "nn_dist_to_normal_core": None,
                    "nn_dist_to_normal_core_edited": None,
                    "z_abs_mean": None,
                    "z_abs_max": None,
                    "z_abs_mean_edited": None,
                    "z_abs_max_edited": None,
                    "robust_z_abs_mean": None,
                    "robust_z_abs_max": None,
                    "robust_z_abs_mean_edited": None,
                    "robust_z_abs_max_edited": None,
                    "mahal_mean": None,
                    "mahal_max": None,
                    "mahal_mean_edited": None,
                    "mahal_max_edited": None,
                    "meta": None,
                    "evals": None,
                    "method": None,
                }
            )
            return out

        # Extract cf
        x_cf = _to_2d(cf_result.get("x_cf")).detach()
        score_cf = _safe_float(cf_result.get("score"))
        meta = cf_result.get("meta", {})
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            meta = {"raw_meta": meta}

        out["meta"] = meta
        out["score_cf"] = score_cf

        # Validity / margin
        if score_cf is not None:
            out["valid"] = bool(score_cf <= thr)
            out["delta_score_to_thr"] = float(score_cf - thr)
            out["margin_to_thr"] = float((thr - score_cf) / (abs(thr) + 1e-12))
        else:
            out["valid"] = False
            out["delta_score_to_thr"] = None
            out["margin_to_thr"] = None

        # Threshold robustness: check if still valid under +/- small threshold shifts
        if score_cf is None:
            out["valid_robust"] = None
        else:
            checks = {}
            for frac in self.cfg.thr_robust_fracs:
                thr_low = thr * (1.0 - frac)
                thr_high = thr * (1.0 + frac)
                checks[f"valid_at_thr_minus_{frac:.0%}"] = bool(score_cf <= thr_low)
                checks[f"valid_at_thr_plus_{frac:.0%}"] = bool(score_cf <= thr_high)
            out["valid_robust"] = checks

        # Delta and masks
        d = x_cf - x0  # (L,F)
        absd = d.abs()

        # Changed masks
        changed_tf = absd > self.cfg.eps_change  # (L,F)
        changed_t = changed_tf.any(dim=1)  # (L,)
        changed_f = changed_tf.any(dim=0)  # (F,)

        # Change structure - time (segments)
        out.update(_segment_stats_from_mask(changed_t))

        # Change structure - feature/cell sparsity
        L, F = x0.shape
        out["n_changed_feat"] = int(changed_f.sum().item())
        out["frac_changed_feat"] = float(changed_f.float().mean().item())
        out["n_changed_cell"] = int(changed_tf.sum().item())
        out["frac_changed_cell"] = float(changed_tf.float().mean().item())

        # Edited span (for edited-only plausibility)
        edited_span = _span_from_mask(changed_t)

        # Proximity (raw)
        out["dist_rmse"] = float(_rmse(x0, x_cf).cpu().item())
        out["dist_mae"] = float(_mae(x0, x_cf).cpu().item())
        out["dist_max_abs"] = float(_max_abs(x0, x_cf).cpu().item())
        out["dist_trimmed_rmse"] = float(
            _trimmed_rmse(d, trim_q=self.cfg.trim_q).detach().cpu().item()
        )
        out["dist_trimmed_mae"] = float(
            _trimmed_mean_abs(d, trim_q=self.cfg.trim_q).detach().cpu().item()
        )

        # Per-feature proximity (raw)
        per_rmse = torch.sqrt(torch.mean((d) ** 2, dim=0) + 1e-12)  # (F,)
        per_mae = torch.mean(torch.abs(d), dim=0)  # (F,)
        out["per_feature_rmse"] = [float(v) for v in per_rmse.detach().cpu().tolist()]
        out["per_feature_mae"] = [float(v) for v in per_mae.detach().cpu().tolist()]

        # Top changed features by per_feature_rmse
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(F)]
        k = min(topk_features, F)
        top_idx = (
            torch.topk(per_rmse, k=k, largest=True).indices.detach().cpu().tolist()
        )
        out["top_changed_features"] = [
            {
                "feature": feature_names[i],
                "rmse": float(per_rmse[i].detach().cpu().item()),
                "mae": float(per_mae[i].detach().cpu().item()),
            }
            for i in top_idx
        ]

        # Proximity (scaled) if scaler available
        if scaler_spec is not None:
            x0s = scaler_spec.transform(x0)
            xcfs = scaler_spec.transform(x_cf)
            ds = xcfs - x0s
            out["dist_rmse_scaled"] = float(_rmse(x0s, xcfs).cpu().item())
            out["dist_mae_scaled"] = float(_mae(x0s, xcfs).cpu().item())
            out["dist_trimmed_rmse_scaled"] = float(
                _trimmed_rmse(ds, trim_q=self.cfg.trim_q).detach().cpu().item()
            )
            out["dist_trimmed_mae_scaled"] = float(
                _trimmed_mean_abs(ds, trim_q=self.cfg.trim_q).detach().cpu().item()
            )
        else:
            out["dist_rmse_scaled"] = None
            out["dist_mae_scaled"] = None
            out["dist_trimmed_rmse_scaled"] = None
            out["dist_trimmed_mae_scaled"] = None

        # Smoothness of edits (raw)
        if d.shape[0] >= 2:
            d1 = _diff1(d)
            out["smooth_l2_d1"] = float(
                torch.sqrt(torch.mean(d1**2) + 1e-12).detach().cpu().item()
            )
            out["tv_l1_d1"] = float(_tv_l1_first_diff(d).detach().cpu().item())
        else:
            out["smooth_l2_d1"] = 0.0
            out["tv_l1_d1"] = 0.0

        if d.shape[0] >= 3:
            d2 = _diff2(d)
            out["smooth_l2_d2"] = float(
                torch.sqrt(torch.mean(d2**2) + 1e-12).detach().cpu().item()
            )
        else:
            out["smooth_l2_d2"] = 0.0

        out["boundary_discontinuity"] = _boundary_discontinuity(d, changed_t)

        # Budget / bookkeeping if present
        out["evals"] = _safe_float(meta.get("evals"))
        out["method"] = meta.get("method", meta.get("strategy", None))

        # Plausibility metrics require normal_core
        if normal_core is None:
            # keep plausibility keys explicit
            out["bounds_violations"] = None
            out["bounds_violation_frac"] = None
            out["bounds_violations_edited"] = None
            out["bounds_violation_frac_edited"] = None
            out["nn_dist_to_normal_core"] = None
            out["nn_dist_to_normal_core_edited"] = None
            out["z_abs_mean"] = None
            out["z_abs_max"] = None
            out["z_abs_mean_edited"] = None
            out["z_abs_max_edited"] = None
            out["robust_z_abs_mean"] = None
            out["robust_z_abs_max"] = None
            out["robust_z_abs_mean_edited"] = None
            out["robust_z_abs_max_edited"] = None
            out["mahal_mean"] = None
            out["mahal_max"] = None
            out["mahal_mean_edited"] = None
            out["mahal_max_edited"] = None
            return out

        nc = _to_3d(normal_core).detach()

        # 1) Bounds violations (whole window + edited span)
        lo, hi = _feature_bounds_from_normal_core(
            nc,
            q_lo=self.cfg.bounds_q_lo,
            q_hi=self.cfg.bounds_q_hi,
        )
        viol = (x_cf < lo) | (x_cf > hi)  # (L,F)
        n_viol = int(viol.sum().item())
        out["bounds_violations"] = n_viol
        out["bounds_violation_frac"] = float(n_viol / max(1, int(x_cf.numel())))

        if edited_span is None:
            out["bounds_violations_edited"] = 0
            out["bounds_violation_frac_edited"] = 0.0
        else:
            t0, t1 = edited_span
            viol_e = viol[t0 : t1 + 1]
            n_viol_e = int(viol_e.sum().item())
            out["bounds_violations_edited"] = n_viol_e
            out["bounds_violation_frac_edited"] = float(
                n_viol_e / max(1, int(viol_e.numel()))
            )

        # 2) Nearest-neighbor distance to NormalCore (whole + edited span)
        out["nn_dist_to_normal_core"] = _nn_plausibility_dist(
            x_cf=x_cf,
            normal_core=nc,
            metric=self.cfg.nn_metric,
            chunk=self.cfg.nn_chunk,
            time_slice=None,
        )
        out["nn_dist_to_normal_core_edited"] = _nn_plausibility_dist(
            x_cf=x_cf,
            normal_core=nc,
            metric=self.cfg.nn_metric,
            chunk=self.cfg.nn_chunk,
            time_slice=edited_span,
        )

        # 3) Z-score plausibility (mean/std) + robust z (MAD), whole + edited span
        z_all = _zscore_plausibility(x_cf=x_cf, normal_core=nc, time_slice=None)
        z_ed = _zscore_plausibility(x_cf=x_cf, normal_core=nc, time_slice=edited_span)
        out["z_abs_mean"] = z_all["z_abs_mean"]
        out["z_abs_max"] = z_all["z_abs_max"]
        out["z_abs_mean_edited"] = z_ed["z_abs_mean"]
        out["z_abs_max_edited"] = z_ed["z_abs_max"]

        rz_all = _robust_z_mad(x_cf=x_cf, normal_core=nc, time_slice=None)
        rz_ed = _robust_z_mad(x_cf=x_cf, normal_core=nc, time_slice=edited_span)
        out["robust_z_abs_mean"] = rz_all["robust_z_abs_mean"]
        out["robust_z_abs_max"] = rz_all["robust_z_abs_max"]
        out["robust_z_abs_mean_edited"] = rz_ed["robust_z_abs_mean"]
        out["robust_z_abs_max_edited"] = rz_ed["robust_z_abs_max"]

        # 4) Correlation-aware plausibility: Mahalanobis, whole + edited span
        mh_all = _mahalanobis_plausibility(x_cf=x_cf, normal_core=nc, time_slice=None)
        mh_ed = _mahalanobis_plausibility(
            x_cf=x_cf, normal_core=nc, time_slice=edited_span
        )
        out["mahal_mean"] = mh_all["mahal_mean"]
        out["mahal_max"] = mh_all["mahal_max"]
        out["mahal_mean_edited"] = mh_ed["mahal_mean"]
        out["mahal_max_edited"] = mh_ed["mahal_max"]

        return out

    def compute_batch(
        self,
        xs: torch.Tensor,
        cf_results: List[Optional[Dict[str, Any]]],
        threshold: float,
        normal_core: Optional[torch.Tensor] = None,
        scaler: Optional[Union[ScalerSpec, Dict[str, Any]]] = None,
        scaler_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        xs3 = _to_3d(xs) if xs.dim() != 2 else xs.unsqueeze(0)
        B = xs3.shape[0]
        if len(cf_results) != B:
            raise ValueError(f"cf_results length {len(cf_results)} != batch size {B}")

        return [
            self.compute(
                xs3[i],
                cf_results[i],
                threshold=threshold,
                normal_core=normal_core,
                scaler=scaler,
                scaler_path=scaler_path,
                feature_names=feature_names,
            )
            for i in range(B)
        ]


# -----------------------------
# Summaries and stability
# -----------------------------
def summarize_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates a list of per-sample metric dicts into summary stats.
    Deterministic; ignores None entries.
    """
    if not metrics:
        return {}

    def collect(key: str) -> List[float]:
        vals: List[float] = []
        for m in metrics:
            v = m.get(key, None)
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            else:
                fv = _safe_float(v)
                if fv is not None:
                    vals.append(float(fv))
        return vals

    def stats(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"mean": None, "median": None, "min": None, "max": None}
        t = torch.tensor(vals, dtype=torch.float64)
        return {
            "mean": float(t.mean().item()),
            "median": float(t.median().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    # keys you likely want in your paper tables
    keys = [
        "found",
        "valid",
        "score_cf",
        "delta_score_to_thr",
        "margin_to_thr",
        "dist_rmse",
        "dist_mae",
        "dist_max_abs",
        "dist_trimmed_rmse",
        "dist_trimmed_mae",
        "dist_rmse_scaled",
        "dist_mae_scaled",
        "dist_trimmed_rmse_scaled",
        "dist_trimmed_mae_scaled",
        "n_changed_time",
        "frac_changed_time",
        "n_segments_time",
        "max_segment_len_time",
        "n_changed_feat",
        "frac_changed_feat",
        "n_changed_cell",
        "frac_changed_cell",
        "smooth_l2_d1",
        "smooth_l2_d2",
        "tv_l1_d1",
        "boundary_discontinuity",
        "bounds_violations",
        "bounds_violation_frac",
        "bounds_violations_edited",
        "bounds_violation_frac_edited",
        "nn_dist_to_normal_core",
        "nn_dist_to_normal_core_edited",
        "z_abs_mean",
        "z_abs_max",
        "z_abs_mean_edited",
        "z_abs_max_edited",
        "robust_z_abs_mean",
        "robust_z_abs_max",
        "robust_z_abs_mean_edited",
        "robust_z_abs_max_edited",
        "mahal_mean",
        "mahal_max",
        "mahal_mean_edited",
        "mahal_max_edited",
        "evals",
    ]

    out: Dict[str, Any] = {
        "n": len(metrics),
        "found_rate": stats(collect("found"))["mean"],
        "valid_rate": stats(collect("valid"))["mean"],
    }

    for k in keys:
        if k in ("found", "valid"):
            continue
        out[k] = stats(collect(k))

    return out


def stability_metrics(
    per_seed_metrics: List[Dict[str, Any]],
    keys: Tuple[str, ...] = (
        "score_cf",
        "margin_to_thr",
        "dist_rmse_scaled",
        "frac_changed_time",
        "frac_changed_feat",
        "n_segments_time",
        "nn_dist_to_normal_core_edited",
        "mahal_mean_edited",
    ),
) -> Dict[str, Any]:
    """
    per_seed_metrics: metrics computed for the SAME x but different seeds / runs.
    Returns variance and range for selected keys. Deterministic.
    """
    out: Dict[str, Any] = {"n_runs": len(per_seed_metrics)}
    for k in keys:
        vals = []
        for m in per_seed_metrics:
            v = _safe_float(m.get(k))
            if v is not None:
                vals.append(v)
        if not vals:
            out[k] = {"var": None, "std": None, "min": None, "max": None, "range": None}
            continue
        t = torch.tensor(vals, dtype=torch.float64)
        var = float(t.var(unbiased=False).item()) if t.numel() > 1 else 0.0
        std = float(torch.sqrt(torch.tensor(var)).item())
        mn = float(t.min().item())
        mx = float(t.max().item())
        out[k] = {"var": var, "std": std, "min": mn, "max": mx, "range": float(mx - mn)}
    return out


def _cmp_value(m: Dict[str, Any], key: str, default: float = float("inf")) -> float:
    v = _safe_float(m.get(key))
    if v is None:
        return default
    return float(v)


def _sample_quality_tuple(
    m: Dict[str, Any],
    score_key: str,
    prox_key: str,
    sparsity_key: str,
    plaus_key: Optional[str],
) -> Tuple[float, float, float, float]:
    valid = bool(m.get("valid", False))
    invalid_flag = 0.0 if valid else 1.0
    score = _cmp_value(m, score_key)
    prox = _cmp_value(m, prox_key)
    sparsity = _cmp_value(m, sparsity_key)
    plaus = _cmp_value(m, plaus_key) if plaus_key else 0.0
    return (invalid_flag, score, prox, sparsity + plaus)


def compare_methods(
    metrics_by_method: Dict[str, List[Dict[str, Any]]],
    *,
    score_key: str = "score_cf",
    proximity_key: str = "dist_rmse",
    sparsity_key: str = "frac_changed_time",
    plausibility_key: Optional[str] = "nn_dist_to_normal_core_edited",
) -> Dict[str, Any]:
    """
    Compare methods on aligned samples using lexicographic pairwise wins.
    """
    if not metrics_by_method:
        return {
            "methods": [],
            "n_samples": 0,
            "per_method_summary": {},
            "wins": {},
            "overall_win_rate": {},
            "ranking": [],
            "comparison_keys": {
                "score_key": score_key,
                "proximity_key": proximity_key,
                "sparsity_key": sparsity_key,
                "plausibility_key": plausibility_key,
            },
        }

    methods = sorted(metrics_by_method.keys())
    lengths = {m: len(metrics_by_method[m]) for m in methods}
    n_samples = min(lengths.values()) if lengths else 0

    per_method_summary = {m: summarize_metrics(metrics_by_method[m]) for m in methods}

    wins: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in methods}
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if i == j:
                wins[mi][mj] = {"wins": 0, "comparisons": 0, "win_rate": None}
                continue

            w = 0
            c = 0
            for k in range(n_samples):
                ai = metrics_by_method[mi][k]
                bj = metrics_by_method[mj][k]
                ta = _sample_quality_tuple(
                    ai, score_key, proximity_key, sparsity_key, plausibility_key
                )
                tb = _sample_quality_tuple(
                    bj, score_key, proximity_key, sparsity_key, plausibility_key
                )
                if ta < tb:
                    w += 1
                c += 1

            wins[mi][mj] = {
                "wins": int(w),
                "comparisons": int(c),
                "win_rate": (float(w / c) if c > 0 else None),
            }

    overall_win_rate: Dict[str, float] = {}
    for mi in methods:
        vals = []
        for mj in methods:
            if mi == mj:
                continue
            wr = wins[mi][mj]["win_rate"]
            if wr is not None:
                vals.append(float(wr))
        overall_win_rate[mi] = float(sum(vals) / len(vals)) if vals else 0.0

    ranking = sorted(
        methods,
        key=lambda m: (
            -overall_win_rate[m],
            -_cmp_value(per_method_summary[m], "valid_rate", default=-1.0),
        ),
    )

    return {
        "methods": methods,
        "n_samples": int(n_samples),
        "per_method_summary": per_method_summary,
        "wins": wins,
        "overall_win_rate": overall_win_rate,
        "ranking": ranking,
        "comparison_keys": {
            "score_key": score_key,
            "proximity_key": proximity_key,
            "sparsity_key": sparsity_key,
            "plausibility_key": plausibility_key,
        },
        "lengths_per_method": lengths,
    }


def estimate_difficulty_score_from_pareto(
    pareto_objectives: List[List[float]],
    threshold: float,
) -> Optional[float]:
    """
    Difficulty proxy from Pareto front:
    sqrt(min proximity objective f2 among points with f1 <= threshold).
    """
    if not pareto_objectives:
        return None

    thr = float(threshold)
    vals: List[float] = []
    for row in pareto_objectives:
        if len(row) < 2:
            continue
        f1 = _safe_float(row[0])
        f2 = _safe_float(row[1])
        if f1 is None or f2 is None:
            continue
        if f1 <= thr:
            vals.append(float(f2))

    if not vals:
        return None
    return float(math.sqrt(max(min(vals), 0.0)))


def genetic_stability_analysis(
    explainer: Any,
    x: Union[np.ndarray, torch.Tensor],
    n_perturbations: int = 8,
    perturb_eps: float = 0.01,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    S(x) style stability: compare CF(x) against CF(x + epsilon) across perturbations.
    """
    if n_perturbations < 1:
        raise ValueError("n_perturbations must be >= 1")

    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    if x_np.ndim != 2:
        raise ValueError(f"x must have shape (L,F), got {x_np.shape}")

    rng = np.random.default_rng(int(random_seed))

    def _extract_x_cf(res: Any) -> Optional[np.ndarray]:
        if res is None:
            return None
        if isinstance(res, dict):
            v = res.get("x_cf")
        else:
            v = getattr(res, "x_cf", None)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"mean": None, "median": None, "min": None, "max": None}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    base_res = explainer.explain(x_np)
    base_cf = _extract_x_cf(base_res)
    if base_cf is None:
        return {
            "n_runs": int(n_perturbations),
            "base_found": False,
            "successful_runs": 0,
            "distance_rmse": {"mean": None, "median": None, "min": None, "max": None},
            "distance_mae": {"mean": None, "median": None, "min": None, "max": None},
            "distances_rmse": [],
            "distances_mae": [],
        }

    dist_rmse: List[float] = []
    dist_mae: List[float] = []
    for _ in range(int(n_perturbations)):
        noise = rng.normal(0.0, float(perturb_eps), size=x_np.shape)
        xp = x_np + noise
        res_p = explainer.explain(xp)
        cf_p = _extract_x_cf(res_p)
        if cf_p is None or cf_p.shape != base_cf.shape:
            continue

        delta = cf_p - base_cf
        dist_rmse.append(float(np.sqrt(np.mean(delta * delta) + 1e-12)))
        dist_mae.append(float(np.mean(np.abs(delta))))

    return {
        "n_runs": int(n_perturbations),
        "base_found": True,
        "successful_runs": int(len(dist_rmse)),
        "distance_rmse": _stats(dist_rmse),
        "distance_mae": _stats(dist_mae),
        "distances_rmse": dist_rmse,
        "distances_mae": dist_mae,
    }
