from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np

from cftsad.core.scoring import compute_threshold_from_normal_core
from cftsad.methods.motif import generate_motif
from cftsad.methods.nearest import generate_nearest
from cftsad.methods.segment import generate_segment
from cftsad.types import CFFailure, CFResult

_ALLOWED_METHODS = {"nearest", "segment", "motif"}


class CounterfactualExplainer:
    def __init__(
        self,
        method: Literal["nearest", "segment", "motif"],
        model,
        normal_core: np.ndarray,
        threshold: Optional[float] = None,
        *,
        immutable_features: Optional[Iterable[int]] = None,
        bounds: Optional[Dict[int, Tuple[float, float]]] = None,
        random_seed: int = 42,
        motif_top_k: int = 5,
        segment_smoothing: bool = False,
    ):
        self.method = str(method)
        self.model = model
        self.normal_core = np.asarray(normal_core)
        self.threshold = threshold
        self.immutable_features = tuple(immutable_features or [])
        self.bounds = bounds or {}
        self.random_seed = int(random_seed)
        self.motif_top_k = int(motif_top_k)
        self.segment_smoothing = bool(segment_smoothing)

        np.random.seed(self.random_seed)

        invalid_reason = self._validate_constructor_inputs()
        if invalid_reason is not None:
            raise ValueError(invalid_reason)

        if self.threshold is None:
            self.threshold = compute_threshold_from_normal_core(self.model, self.normal_core, quantile=0.95)
        else:
            self.threshold = float(self.threshold)

    def _validate_constructor_inputs(self) -> Optional[str]:
        if self.method not in _ALLOWED_METHODS:
            return f"method must be one of {_ALLOWED_METHODS}, got {self.method!r}"

        if self.normal_core.ndim != 3:
            return "normal_core must have shape (K, L, F)"

        if self.normal_core.shape[0] == 0:
            return "normal_core must contain at least one window"

        if np.isnan(self.normal_core).any():
            return "normal_core contains NaN values; v1 does not allow NaNs"

        if self.threshold is not None:
            try:
                thr = float(self.threshold)
            except Exception:
                return "threshold must be a numeric scalar"
            if not np.isfinite(thr) or thr < 0.0:
                return "threshold must be finite and non-negative"

        _, _, n_features = self.normal_core.shape
        for feat_idx in self.immutable_features:
            idx = int(feat_idx)
            if idx < 0 or idx >= n_features:
                return f"immutable feature index {idx} out of range for F={n_features}"

        for feat_idx, pair in self.bounds.items():
            idx = int(feat_idx)
            if idx < 0 or idx >= n_features:
                return f"bounds feature index {idx} out of range for F={n_features}"
            if not isinstance(pair, tuple) or len(pair) != 2:
                return f"bounds for feature {idx} must be a (min_val, max_val) tuple"
            lo, hi = float(pair[0]), float(pair[1])
            if lo > hi:
                return f"bounds for feature {idx} invalid: min_val > max_val"

        return None

    def _validate_x(self, x: np.ndarray) -> Optional[str]:
        x = np.asarray(x)
        if x.ndim != 2:
            return "x must have shape (L, F)"

        L, F = x.shape
        _, core_L, core_F = self.normal_core.shape
        if (L, F) != (core_L, core_F):
            return f"x shape mismatch: expected {(core_L, core_F)}, got {(L, F)}"

        if np.isnan(x).any():
            return "x contains NaN values; v1 does not allow NaNs"

        return None

    def explain(self, x: np.ndarray) -> CFResult | CFFailure:
        x_arr = np.asarray(x, dtype=np.float64)
        invalid = self._validate_x(x_arr)
        if invalid is not None:
            return CFFailure(
                reason="invalid_input",
                message=invalid,
                diagnostics={
                    "x_shape": tuple(x_arr.shape),
                    "normal_core_shape": tuple(self.normal_core.shape),
                },
            )

        try:
            if self.method == "nearest":
                return generate_nearest(
                    model=self.model,
                    x=x_arr,
                    normal_core=self.normal_core,
                    threshold=float(self.threshold),
                    immutable_features=self.immutable_features,
                    bounds=self.bounds,
                )
            if self.method == "segment":
                return generate_segment(
                    model=self.model,
                    x=x_arr,
                    normal_core=self.normal_core,
                    threshold=float(self.threshold),
                    immutable_features=self.immutable_features,
                    bounds=self.bounds,
                    smoothing=self.segment_smoothing,
                )
            if self.method == "motif":
                return generate_motif(
                    model=self.model,
                    x=x_arr,
                    normal_core=self.normal_core,
                    threshold=float(self.threshold),
                    immutable_features=self.immutable_features,
                    bounds=self.bounds,
                    top_k=self.motif_top_k,
                )

            return CFFailure(
                reason="invalid_input",
                message=f"Unknown method {self.method!r}",
                diagnostics={},
            )
        except Exception as exc:
            return CFFailure(
                reason="invalid_input",
                message=f"Counterfactual generation failed: {exc}",
                diagnostics={"exception_type": type(exc).__name__},
            )
