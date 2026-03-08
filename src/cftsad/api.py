from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np

from cftsad.core.normal_core import build_normal_core
from cftsad.core.persistence import load_core_artifacts, save_core_artifacts
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
        self.score_fn = method_kwargs.pop("score_fn", None)
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
        self.normal_core_embedding_dim = method_kwargs.pop(
            "normal_core_embedding_dim", 32
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
        self.use_plausibility_objective = bool(
            method_kwargs.pop("use_plausibility_objective", True)
        )
        self.structured_mutation_weight = float(
            method_kwargs.pop("structured_mutation_weight", 0.35)
        )
        self.validity_margin = float(method_kwargs.pop("validity_margin", 0.0))
        self.top_m_solutions = int(method_kwargs.pop("top_m_solutions", 5))
        self.early_stop_patience = int(method_kwargs.pop("early_stop_patience", 15))
        self.use_constraints_v2 = bool(method_kwargs.pop("use_constraints_v2", True))
        self.max_delta_per_step = method_kwargs.pop("max_delta_per_step", None)
        self.relational_linear = method_kwargs.pop("relational_linear", None)

        self.enable_fallback_chain = bool(
            method_kwargs.pop("enable_fallback_chain", True)
        )
        self.fallback_methods = tuple(
            str(v)
            for v in method_kwargs.pop(
                "fallback_methods",
                ("segment", "motif", "nearest"),
            )
        )
        self.fallback_retry_budget = int(method_kwargs.pop("fallback_retry_budget", 3))

        self.method_kwargs = method_kwargs

        np.random.seed(self.random_seed)

        invalid_reason = self._validate_constructor_inputs()
        if invalid_reason is not None:
            raise ValueError(invalid_reason)

        core_build = build_normal_core(
            model=self.model,
            normal_core=self.normal_core,
            threshold=self.threshold,
            score_fn=self.score_fn,
            filter_factor=self.normal_core_filter_factor,
            threshold_quantile=self.normal_core_threshold_quantile,
            max_core_size=self.normal_core_max_size,
            embedding_dim=self.normal_core_embedding_dim,
            use_diversity_sampling=self.normal_core_use_diversity_sampling,
            random_seed=self.random_seed,
        )
        self.normal_core = core_build.normal_core
        self.core_index = core_build.selected_indices
        self.core_embeddings = core_build.embeddings
        self.core_reduced_embeddings = core_build.reduced_embeddings
        self.core_pca_components = core_build.pca_components
        self.core_pca_mean = core_build.pca_mean
        self.core_scores = core_build.selected_scores
        self.core_scores_all = core_build.all_scores
        self.core_build_info = {
            "base_threshold": float(core_build.base_threshold),
            "strict_threshold": float(core_build.strict_threshold),
            "core_size_before": int(self.core_scores_all.shape[0]),
            "core_size_after": int(self.normal_core.shape[0]),
            "embedding_dim": int(self.core_reduced_embeddings.shape[1]),
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

        if self.score_fn is not None and not callable(self.score_fn):
            return "score_fn must be callable when provided"

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

        if self.normal_core_embedding_dim is not None:
            try:
                emb_dim = int(self.normal_core_embedding_dim)
            except Exception:
                return "normal_core_embedding_dim must be an integer when provided"
            if emb_dim < 1:
                return "normal_core_embedding_dim must be >= 1 when provided"
            self.normal_core_embedding_dim = emb_dim

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
        if not (0.0 <= self.structured_mutation_weight <= 1.0):
            return "structured_mutation_weight must be in [0, 1]"
        if not np.isfinite(self.validity_margin) or self.validity_margin < 0.0:
            return "validity_margin must be finite and >= 0"
        if self.top_m_solutions < 1:
            return "top_m_solutions must be >= 1"
        if self.early_stop_patience < 1:
            return "early_stop_patience must be >= 1"
        if self.max_delta_per_step is not None:
            try:
                max_delta = float(self.max_delta_per_step)
            except Exception:
                return "max_delta_per_step must be numeric when provided"
            if not np.isfinite(max_delta) or max_delta <= 0.0:
                return "max_delta_per_step must be finite and > 0 when provided"
            self.max_delta_per_step = max_delta
        if self.relational_linear is not None and not isinstance(
            self.relational_linear, dict
        ):
            return "relational_linear must be a dict when provided"

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

        if self.fallback_retry_budget < 0:
            return "fallback_retry_budget must be >= 0"
        for m in self.fallback_methods:
            if m not in _ALLOWED_METHODS:
                return f"fallback method must be one of {_ALLOWED_METHODS}, got {m!r}"

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

    def _run_method(self, method: str, x_arr: np.ndarray) -> CFResult | CFFailure:
        if method == "nearest":
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
                use_constraints_v2=self.use_constraints_v2,
                max_delta_per_step=self.max_delta_per_step,
                relational_linear=self.relational_linear,
                score_fn=self.score_fn,
            )

        if method == "segment":
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
                use_constraints_v2=self.use_constraints_v2,
                max_delta_per_step=self.max_delta_per_step,
                relational_linear=self.relational_linear,
                score_fn=self.score_fn,
            )

        if method == "motif":
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
                use_constraints_v2=self.use_constraints_v2,
                max_delta_per_step=self.max_delta_per_step,
                relational_linear=self.relational_linear,
                score_fn=self.score_fn,
            )

        if method == "genetic":
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
                use_plausibility_objective=self.use_plausibility_objective,
                structured_mutation_weight=self.structured_mutation_weight,
                validity_margin=self.validity_margin,
                top_m_solutions=self.top_m_solutions,
                early_stop_patience=self.early_stop_patience,
                use_constraints_v2=self.use_constraints_v2,
                max_delta_per_step=self.max_delta_per_step,
                relational_linear=self.relational_linear,
                score_fn=self.score_fn,
                random_seed=self.random_seed,
            )

        return CFFailure(
            reason="invalid_input",
            message=f"Unknown method {method!r}",
            diagnostics={},
        )

    def save_core(self, path: str) -> None:
        save_core_artifacts(
            path=path,
            normal_core=self.normal_core,
            selected_indices=self.core_index,
            embeddings=self.core_embeddings,
            reduced_embeddings=self.core_reduced_embeddings,
            pca_components=self.core_pca_components,
            pca_mean=self.core_pca_mean,
            selected_scores=self.core_scores,
            all_scores=self.core_scores_all,
            metadata=self.core_build_info,
        )

    @staticmethod
    def load_core(path: str) -> Dict[str, object]:
        return load_core_artifacts(path)

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
            primary = self._run_method(self.method, x_arr)
            if isinstance(primary, CFResult):
                return primary

            if not self.enable_fallback_chain:
                return primary

            eligible = {"no_valid_cf", "segment_detection_failed", "optimization_failed"}
            if primary.reason not in eligible:
                return primary

            attempts = []
            budget = int(self.fallback_retry_budget)
            if budget == 0:
                primary.diagnostics["fallback_attempts"] = attempts
                return primary

            for method in self.fallback_methods:
                if method == self.method:
                    continue
                if len(attempts) >= budget:
                    break
                try:
                    result = self._run_method(method, x_arr)
                except Exception as exc:
                    attempts.append(
                        {
                            "method": method,
                            "status": "error",
                            "reason": type(exc).__name__,
                        }
                    )
                    continue

                if isinstance(result, CFResult):
                    result.meta["fallback_used"] = True
                    result.meta["fallback_from"] = self.method
                    result.meta["fallback_method"] = method
                    result.meta["fallback_attempts"] = attempts
                    return result

                attempts.append(
                    {
                        "method": method,
                        "status": "failed",
                        "reason": result.reason,
                    }
                )

            primary.diagnostics["fallback_attempts"] = attempts
            return primary
        except Exception as exc:
            return CFFailure(
                reason="optimization_failed"
                if self.method == "genetic"
                else "invalid_input",
                message=f"Counterfactual generation failed: {exc}",
                diagnostics={"exception_type": type(exc).__name__},
            )
