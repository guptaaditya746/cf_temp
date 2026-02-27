from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np

from cftsad.core.normal_core import build_normal_core
from cftsad.methods.genetic import generate_genetic
from cftsad.methods.motif import generate_motif
from cftsad.methods.nearest import generate_nearest
from cftsad.methods.segment import generate_segment
from cftsad.types import CFFailure, CFResult

_ALLOWED_METHODS = {"nearest", "segment", "motif", "genetic"}


class CounterfactualExplainer:
    def __init__(
        self,
        method: Literal["nearest", "segment", "motif", "genetic"],
        model,
        normal_core: np.ndarray,
        threshold: Optional[float] = None,
        **method_kwargs,
    ):
        self.method = str(method)
        self.model = model
        self.normal_core = np.asarray(normal_core)
        self.threshold = threshold

        # Common knobs (usable by all methods)
        self.immutable_features = tuple(method_kwargs.pop("immutable_features", ()))
        self.bounds = dict(method_kwargs.pop("bounds", {}))
        self.random_seed = int(method_kwargs.pop("random_seed", 42))
        self.normal_core_filter_factor = float(
            method_kwargs.pop("normal_core_filter_factor", 1.0)
        )
        self.normal_core_threshold_quantile = float(
            method_kwargs.pop("normal_core_threshold_quantile", 0.95)
        )
        self.normal_core_max_size = method_kwargs.pop("normal_core_max_size", None)
        self.normal_core_use_diversity_sampling = bool(
            method_kwargs.pop("normal_core_use_diversity_sampling", True)
        )

        # Method-specific knobs
        self.motif_top_k = int(method_kwargs.pop("motif_top_k", 5))
        self.motif_n_segments = int(method_kwargs.pop("motif_n_segments", 4))
        self.motif_length_factors = tuple(
            float(v) for v in method_kwargs.pop("motif_length_factors", (0.75, 1.0, 1.25))
        )
        self.motif_context_weight = float(method_kwargs.pop("motif_context_weight", 0.2))
        self.motif_use_affine_fit = bool(method_kwargs.pop("motif_use_affine_fit", True))

        self.segment_smoothing = bool(method_kwargs.pop("segment_smoothing", False))
        self.segment_n_candidates = int(method_kwargs.pop("segment_n_candidates", 4))
        self.segment_top_k_donors = int(method_kwargs.pop("segment_top_k_donors", 8))
        self.segment_context_width = int(method_kwargs.pop("segment_context_width", 2))
        self.segment_crossfade_width = int(method_kwargs.pop("segment_crossfade_width", 3))

        self.nearest_top_k = int(method_kwargs.pop("nearest_top_k", 10))
        self.nearest_alpha_steps = int(method_kwargs.pop("nearest_alpha_steps", 11))
        self.nearest_donor_filter_factor = float(
            method_kwargs.pop("nearest_donor_filter_factor", 1.0)
        )
        self.nearest_use_weighted_distance = bool(
            method_kwargs.pop("nearest_use_weighted_distance", True)
        )

        self.population_size = int(method_kwargs.pop("population_size", 100))
        self.n_generations = int(method_kwargs.pop("n_generations", 50))
        self.crossover_rate = float(method_kwargs.pop("crossover_rate", 0.9))
        self.mutation_rate = float(method_kwargs.pop("mutation_rate", 0.1))
        self.mutation_sigma = float(method_kwargs.pop("mutation_sigma", 0.05))
        self.use_smoothness_objective = bool(
            method_kwargs.pop("use_smoothness_objective", False)
        )

        self.method_kwargs = method_kwargs

        np.random.seed(self.random_seed)

        invalid_reason = self._validate_constructor_inputs()
        if invalid_reason is not None:
            raise ValueError(invalid_reason)

        core_build = build_normal_core(
            model=self.model,
            normal_core=self.normal_core,
            threshold=self.threshold,
            filter_factor=self.normal_core_filter_factor,
            threshold_quantile=self.normal_core_threshold_quantile,
            max_core_size=self.normal_core_max_size,
            use_diversity_sampling=self.normal_core_use_diversity_sampling,
            random_seed=self.random_seed,
        )
        self.normal_core = core_build.normal_core
        self.core_index = core_build.selected_indices
        self.core_embeddings = core_build.embeddings
        self.core_scores = core_build.selected_scores
        self.core_scores_all = core_build.all_scores
        self.core_build_info = {
            "base_threshold": float(core_build.base_threshold),
            "strict_threshold": float(core_build.strict_threshold),
            "core_size_before": int(self.core_scores_all.shape[0]),
            "core_size_after": int(self.normal_core.shape[0]),
        }

        if self.threshold is None:
            self.threshold = float(core_build.base_threshold)
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
            return "normal_core contains NaN values; v2.0 does not allow NaNs"

        if self.threshold is not None:
            try:
                thr = float(self.threshold)
            except Exception:
                return "threshold must be a numeric scalar"
            if not np.isfinite(thr) or thr < 0.0:
                return "threshold must be finite and non-negative"

        if (
            not np.isfinite(self.normal_core_filter_factor)
            or self.normal_core_filter_factor <= 0.0
        ):
            return "normal_core_filter_factor must be finite and > 0"

        if not (0.0 < self.normal_core_threshold_quantile < 1.0):
            return "normal_core_threshold_quantile must be in (0, 1)"

        if self.normal_core_max_size is not None:
            try:
                max_size = int(self.normal_core_max_size)
            except Exception:
                return "normal_core_max_size must be an integer when provided"
            if max_size < 1:
                return "normal_core_max_size must be >= 1 when provided"
            self.normal_core_max_size = max_size

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

        if self.population_size < 4:
            return "population_size must be >= 4"
        if self.n_generations < 1:
            return "n_generations must be >= 1"
        if not (0.0 <= self.crossover_rate <= 1.0):
            return "crossover_rate must be in [0, 1]"
        if not (0.0 <= self.mutation_rate <= 1.0):
            return "mutation_rate must be in [0, 1]"
        if not (self.mutation_sigma > 0.0):
            return "mutation_sigma must be > 0"

        if self.nearest_top_k < 1:
            return "nearest_top_k must be >= 1"
        if self.nearest_alpha_steps < 2:
            return "nearest_alpha_steps must be >= 2"
        if (
            not np.isfinite(self.nearest_donor_filter_factor)
            or self.nearest_donor_filter_factor <= 0.0
        ):
            return "nearest_donor_filter_factor must be finite and > 0"

        if self.segment_n_candidates < 1:
            return "segment_n_candidates must be >= 1"
        if self.segment_top_k_donors < 1:
            return "segment_top_k_donors must be >= 1"
        if self.segment_context_width < 0:
            return "segment_context_width must be >= 0"
        if self.segment_crossfade_width < 1:
            return "segment_crossfade_width must be >= 1"

        if self.motif_n_segments < 1:
            return "motif_n_segments must be >= 1"
        if self.motif_top_k < 1:
            return "motif_top_k must be >= 1"
        if not self.motif_length_factors:
            return "motif_length_factors must be non-empty"
        if any(v <= 0.0 or not np.isfinite(v) for v in self.motif_length_factors):
            return "motif_length_factors values must be finite and > 0"
        if (
            not np.isfinite(self.motif_context_weight)
            or self.motif_context_weight < 0.0
        ):
            return "motif_context_weight must be finite and >= 0"

        if self.method_kwargs:
            bad = ", ".join(sorted(self.method_kwargs.keys()))
            return f"unknown method kwargs: {bad}"

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
            return "x contains NaN values; v2.0 does not allow NaNs"

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
                    top_k=self.nearest_top_k,
                    alpha_steps=self.nearest_alpha_steps,
                    donor_filter_factor=self.nearest_donor_filter_factor,
                    use_weighted_distance=self.nearest_use_weighted_distance,
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
                    n_segments=self.segment_n_candidates,
                    top_k_donors=self.segment_top_k_donors,
                    context_width=self.segment_context_width,
                    crossfade_width=self.segment_crossfade_width,
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
                    n_segments=self.motif_n_segments,
                    length_factors=self.motif_length_factors,
                    context_weight=self.motif_context_weight,
                    use_affine_fit=self.motif_use_affine_fit,
                )

            if self.method == "genetic":
                return generate_genetic(
                    model=self.model,
                    x=x_arr,
                    normal_core=self.normal_core,
                    threshold=float(self.threshold),
                    immutable_features=self.immutable_features,
                    bounds=self.bounds,
                    population_size=self.population_size,
                    n_generations=self.n_generations,
                    crossover_rate=self.crossover_rate,
                    mutation_rate=self.mutation_rate,
                    mutation_sigma=self.mutation_sigma,
                    use_smoothness_objective=self.use_smoothness_objective,
                    random_seed=self.random_seed,
                )

            return CFFailure(
                reason="invalid_input",
                message=f"Unknown method {self.method!r}",
                diagnostics={},
            )
        except Exception as exc:
            return CFFailure(
                reason="optimization_failed"
                if self.method == "genetic"
                else "invalid_input",
                message=f"Counterfactual generation failed: {exc}",
                diagnostics={"exception_type": type(exc).__name__},
            )
