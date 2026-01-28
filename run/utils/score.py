# score.py
# Centralized scoring utilities for reconstruction-based anomaly detection
# - single source of truth used by: calibration, inference, error_maps, CF methods
#
# Supports numpy OR torch inputs.
# Shapes supported:
#   x, x_hat: (L,F) or (B,L,F)
# Returns are always numpy arrays (for easy saving to .npz), unless return_torch=True.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  # torch optional for pure-numpy usage
    torch = None

ArrayLike = Union[np.ndarray, "torch.Tensor"]
Mode = Literal["mse", "mae"]
Agg = Literal["mean", "sum", "max", "p95", "p99"]


def _is_torch(x: ArrayLike) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_3d(x: ArrayLike) -> Tuple[ArrayLike, bool]:
    """
    Ensure shape is (B,L,F). If input is (L,F), adds batch dim.
    Returns (x_3d, added_batch_dim_flag)
    """
    if _is_torch(x):
        if x.ndim == 2:
            return x.unsqueeze(0), True
        if x.ndim == 3:
            return x, False
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")
    else:
        x = np.asarray(x)
        if x.ndim == 2:
            return x[None, ...], True
        if x.ndim == 3:
            return x, False
        raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")


def _reduce(E: ArrayLike, axis: Tuple[int, ...], agg: Agg) -> ArrayLike:
    """
    Reduce E over axis with aggregation agg.
    """
    if _is_torch(E):
        if agg == "mean":
            return E.mean(dim=axis)
        if agg == "sum":
            return E.sum(dim=axis)
        if agg == "max":
            return E.amax(dim=axis)
        if agg in ("p95", "p99"):
            q = 0.95 if agg == "p95" else 0.99
            # torch.quantile supports reducing over specified dim in newer versions;
            # if not available, fallback to numpy conversion (still deterministic)
            if hasattr(torch, "quantile"):
                return torch.quantile(E, q=q, dim=axis)
            En = _to_numpy(E)
            return np.quantile(En, q=q, axis=axis)
        raise ValueError(f"Unknown agg: {agg}")
    else:
        if agg == "mean":
            return E.mean(axis=axis)
        if agg == "sum":
            return E.sum(axis=axis)
        if agg == "max":
            return E.max(axis=axis)
        if agg == "p95":
            return np.quantile(E, 0.95, axis=axis)
        if agg == "p99":
            return np.quantile(E, 0.99, axis=axis)
        raise ValueError(f"Unknown agg: {agg}")


# ============================================================================
# 1) error_map(x, x_hat)
# ============================================================================
def error_map(
    x: ArrayLike,
    x_hat: ArrayLike,
    mode: Mode = "mse",
    return_torch: bool = False,
) -> ArrayLike:
    """
    Compute per-time per-feature error map E.
    Inputs:
      x, x_hat: (L,F) or (B,L,F)
    Returns:
      E: (B,L,F) or (L,F) depending on input, unless return_torch=True forces torch
    """
    x3, added_x = _ensure_3d(x)
    h3, added_h = _ensure_3d(x_hat)

    if added_x != added_h:
        raise ValueError("x and x_hat must have matching batch dimensionality")

    if _is_torch(x3) != _is_torch(h3):
        # mixed inputs: convert both to numpy
        x3 = _to_numpy(x3)
        h3 = _to_numpy(h3)

    if mode == "mse":
        E = (x3 - h3) ** 2
    elif mode == "mae":
        E = (x3 - h3).abs() if _is_torch(x3) else np.abs(x3 - h3)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Remove batch dim if it was originally absent
    if added_x:
        E2 = E[0]
    else:
        E2 = E

    if return_torch and not _is_torch(E2):
        if torch is None:
            raise RuntimeError("return_torch=True requires torch installed")
        return torch.from_numpy(np.asarray(E2)).float()
    if (not return_torch) and _is_torch(E2):
        return _to_numpy(E2)
    return E2


# ============================================================================
# 2) time_error(E) -> (B,L) or (L,)
# ============================================================================
def time_error(
    E: ArrayLike,
    agg: Agg = "mean",
    return_numpy: bool = True,
) -> np.ndarray:
    """
    Aggregate error map over features to get per-timestep error.
    E: (L,F) or (B,L,F)
    Returns numpy: (L,) or (B,L)
    """
    E3, added = _ensure_3d(E)
    et = _reduce(E3, axis=(2,), agg=agg)  # reduce over F
    if added:
        et = et[0]
    return _to_numpy(et) if return_numpy else et


# ============================================================================
# 3) feature_error(E) -> (B,F) or (F,)
# ============================================================================
def feature_error(
    E: ArrayLike,
    agg: Agg = "mean",
    return_numpy: bool = True,
) -> np.ndarray:
    """
    Aggregate error map over time to get per-feature error.
    E: (L,F) or (B,L,F)
    Returns numpy: (F,) or (B,F)
    """
    E3, added = _ensure_3d(E)
    ef = _reduce(E3, axis=(1,), agg=agg)  # reduce over L
    if added:
        ef = ef[0]
    return _to_numpy(ef) if return_numpy else ef


# ============================================================================
# 4) window_score(E) -> (B,) or scalar
# ============================================================================
def window_score(
    E: ArrayLike,
    agg: Agg = "mean",
    return_numpy: bool = True,
) -> Union[np.ndarray, float]:
    """
    Aggregate error map over time and features to get scalar anomaly score.
    E: (L,F) or (B,L,F)
    Returns numpy (B,) or scalar float if (L,F).
    """
    E3, added = _ensure_3d(E)
    s = _reduce(E3, axis=(1, 2), agg=agg)  # reduce over L,F
    if added:
        s = s[0]
        out = float(_to_numpy(s))
        return out
    out = _to_numpy(s) if return_numpy else s
    return out


# ============================================================================
# 5) reconstruction_score(x, x_hat) -> scalar score
# ============================================================================
def reconstruction_score(
    x: ArrayLike,
    x_hat: ArrayLike,
    mode: Mode = "mse",
    agg: Agg = "mean",
) -> Union[float, np.ndarray]:
    """
    Convenience: compute error_map then window_score.
    If x is (L,F) -> returns float
    If x is (B,L,F) -> returns (B,) numpy
    """
    E = error_map(x, x_hat, mode=mode, return_torch=False)
    return window_score(E, agg=agg, return_numpy=True)


# ============================================================================
# 6) score_bundle(x, x_hat) -> everything needed for error_maps.npz
# ============================================================================
@dataclass
class ScoreBundle:
    error_map: np.ndarray  # (B,L,F) or (L,F)
    error_time: np.ndarray  # (B,L) or (L,)
    error_feature: np.ndarray  # (B,F) or (F,)
    score: Union[float, np.ndarray]  # scalar or (B,)


def score_bundle(
    x: ArrayLike,
    x_hat: ArrayLike,
    mode: Mode = "mse",
    time_agg: Agg = "mean",
    feat_agg: Agg = "mean",
    score_agg: Agg = "mean",
) -> ScoreBundle:
    """
    One-call utility to produce:
    - error_map (L,F)
    - time_error (L,)
    - feature_error (F,)
    - window score (scalar)
    Suitable for writing error_maps.npz consistently.
    """
    E = error_map(x, x_hat, mode=mode, return_torch=False)
    et = time_error(E, agg=time_agg, return_numpy=True)
    ef = feature_error(E, agg=feat_agg, return_numpy=True)
    sc = window_score(E, agg=score_agg, return_numpy=True)
    return ScoreBundle(
        error_map=np.asarray(E), error_time=et, error_feature=ef, score=sc
    )


# ----------------------------------------------------------------------------
# Minimal self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    L, F = 8, 3
    x = np.random.randn(L, F).astype(np.float32)
    x_hat = x + 0.1 * np.random.randn(L, F).astype(np.float32)

    b = score_bundle(x, x_hat, mode="mse")
    print("E shape:", b.error_map.shape)
    print("e_t shape:", b.error_time.shape)
    print("e_f shape:", b.error_feature.shape)
    print("score:", b.score)
