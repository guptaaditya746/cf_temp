import numpy as np
from scipy.stats import multivariate_normal as mvn


from cftsad.core.postprocess import build_explainability_meta, build_score_summary
from cftsad.core.scoring import reconstruction_errors_per_timestep
from cftsad.types import CFFailure, CFResult


def _stabilize_covariance(cov, min_eig=1e-8):
    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, float(min_eig))
    return (eigvecs * eigvals[None, :]) @ eigvecs.T


def _crossfade_boundaries(x_cf, x_original, start, end, width=2):
    out = np.array(x_cf, copy=True)
    w = max(0, int(width))
    if w == 0:
        return out

    for i in range(1, w + 1):
        li = start - i
        if li >= 0:
            alpha = float(i) / float(w + 1)
            out[li] = (1.0 - alpha) * x_original[li] + alpha * out[li]

        ri = end + i - 1
        if ri < out.shape[0]:
            alpha = float(i) / float(w + 1)
            out[ri] = (1.0 - alpha) * x_original[ri] + alpha * out[ri]

    return out


def _interval_summary(start, end, window_length, errors):
    interval_errors = np.asarray(errors[start:end], dtype=np.float64)
    top_k = min(5, len(errors))
    top_idx = np.argsort(-np.asarray(errors, dtype=np.float64))[:top_k]
    return {
        "interval_start": int(start),
        "interval_end": int(end),
        "interval_length": int(end - start),
        "interval_fraction": float((end - start) / max(1, window_length)),
        "touches_left_boundary": bool(start == 0),
        "touches_right_boundary": bool(end == window_length),
        "peak_error_timestep": int(np.argmax(errors)),
        "peak_error_value": float(np.max(errors)),
        "interval_mean_error": (
            None if interval_errors.size == 0 else float(np.mean(interval_errors))
        ),
        "interval_max_error": (
            None if interval_errors.size == 0 else float(np.max(interval_errors))
        ),
        "top_error_timesteps": [int(i) for i in top_idx.tolist()],
    }


def sample_replacement(X, intvl, td=1, cond=None, enforce_psd=True):
    """ Samples a replacement for a given interval, taking inter-variable and inter-temporal correlations into account.

    Parameters
    ----------
    X : d-times-t array
        The full time series.
    intvl : 2-tuple of int
        First point within and first point after the interval to be replaced.
    td : int, default: 1
        The time-delay embedding dimension used for anomaly detection.
        This controls the context size before and after the interval to be replaced,
        on which the replacement will be conditioned.
    cond : tuple of int, optional
        Indices of observed variables to condition on.
    enforce_psd : bool, default: True
        If True, the estimated covariance matrix will be enforced to be positive semi-definite.
        If False, a warning will be yielded for matrices that are not PSD.

    Returns
    -------
    replacement : 2-d array
        The sampled replacement for the given interval.
        The length of the replacement (2nd dimension) equals the length of the interval.
        The number of variables (1st dimension) equals the original number of variables minus
        the number of observed variables.
    """

    assert(X.ndim == 2)
    assert(td > 0)

    dim = X.shape[0]
    length = intvl[1] - intvl[0]

    # Replace outer interval with missing values
    outer = X.copy()
    outer[:, intvl[0]:intvl[1]] = np.nan

    # Compute mean
    mean = np.nanmean(outer, axis=1)
    rep_mean = np.tile(mean, (length + 2 * td - 2,))

    # Compute inter-variable and inter-temporal covariance (Block Toeplitz Matrix)
    centered = outer - mean[:, None]
    cov = np.zeros((len(rep_mean), len(rep_mean)))
    for t in range(length + 2 * td - 2):
        for v1 in range(dim):
            for v2 in range(dim):
                corr = centered[v1, t:] * centered[v2, :centered.shape[1]-t]
                tv_cov = np.nansum(corr) / (np.sum(np.isfinite(corr)) - 1)
                cov[np.arange(v1, cov.shape[0] - t * dim, dim), np.arange(t * dim + v2, cov.shape[1], dim)] = tv_cov
                cov[np.arange(t * dim + v2, cov.shape[1], dim), np.arange(v1, cov.shape[0] - t * dim, dim)] = tv_cov

    assert(np.allclose(cov, cov.T))

    # Enforce covariance matrix to be PSD
    if enforce_psd:
        cov = _stabilize_covariance(cov)

    # Condition on observed variables and left and right context
    if td > 1:
        context_conditions = np.concatenate([np.arange((td - 1) * dim), np.arange((length + td - 1) * dim, len(rep_mean))])
        variable_conditions = np.concatenate([np.arange(d, len(rep_mean), dim) for d in cond]) \
                              if cond is not None and len(cond) > 0 \
                              else np.zeros((0,), dtype=int)

        # Build a fixed-size context window and pad missing boundary context with NaNs.
        # conditional_mvn will marginalize those missing observations automatically.
        context_start = intvl[0] - td + 1
        context_end = intvl[1] + td - 1
        context_window = np.full((length + 2 * td - 2, dim), np.nan, dtype=float)
        overlap_start = max(0, context_start)
        overlap_end = min(X.shape[1], context_end)
        if overlap_start < overlap_end:
            dest_start = overlap_start - context_start
            dest_end = dest_start + (overlap_end - overlap_start)
            context_window[dest_start:dest_end, :] = X[:, overlap_start:overlap_end].T

        rep_mean, cov = conditional_mvn(
            rep_mean, cov,
            context_window.ravel(),
            np.union1d(context_conditions, variable_conditions)
        )

    # Sample replacement
    if enforce_psd:
        cov = _stabilize_covariance(cov)
    return mvn.rvs(rep_mean, cov).reshape(length, dim - (len(cond) if cond is not None else 0)).T


def conditional_mvn(mu, S, X, d_obs):
    """ Computes a conditional normal distribution.

    Parameters
    ----------
    mu : length-n vector or n-times-t array
        Mean of the unconditional distribution.
        If a 2-d array is given, multiple samples will be drawn with different means and observations.
    S : n-times-n array
        Covariance matrix of the unconditional distribution.
    X : length-n vector or n-times-t array
        Original values of all variables.
        If a 2-d array is given, multiple samples will be drawn with different observations.
    d_obs : list or tuple of int
        Indices of the observed variables in X to condition on.

    Returns
    -------
    mu_cond : 1-d or 2-d array
        Conditional mean.
    S_cond : 2-d array
        Conditional covariance.

    The number of variables of the conditional distribution equals the original number of variables
    minus the number of observed variables.
    """

    assert(S.ndim == 2)
    assert(1 <= mu.ndim <= 2)
    assert(1 <= X.ndim <= 2)
    assert(mu.shape[0] == S.shape[0])
    assert(S.shape[0] == S.shape[1])

    # Obtain indices of variables to be inferred (not observed)
    d_inf = np.setdiff1d(np.arange(len(mu)), d_obs)

    # Drop observed variables that are actually not observed (missing values)
    # (Dropping them is theoretically equivalent to marginalization)
    missing_variables = np.where(np.isnan(X[d_obs, ...]) if X.ndim == 1 else np.any(np.isnan(X[d_obs, ...]), axis=1))[0]
    if len(missing_variables) > 0:
        d_obs = [d_obs[i] for i in range(len(d_obs)) if i not in missing_variables]

    if len(d_obs) == 0:
        d_inf = np.setdiff1d(np.arange(len(mu)), d_obs)
        return mu[d_inf, ...], S[np.ix_(d_inf, d_inf)]

    # Partition covariance matrix
    S11 = S[np.ix_(d_inf, d_inf)]
    S12 = S[np.ix_(d_inf, d_obs)]
    S22 = S[np.ix_(d_obs, d_obs)]

    # Compute conditional mean and covariance
    if X.ndim > 1 and mu.ndim == 1:
        mu = mu[:, None]
    solve_term = np.linalg.pinv(S22)
    mu_cond = mu[d_inf, ...] + S12 @ solve_term @ (X[d_obs, ...] - mu[d_obs, ...])
    S_cond = S11 - S12 @ solve_term @ S12.T
    S_cond = _stabilize_covariance(S_cond)
    return mu_cond, S_cond


def detect_anomalous_interval(x, model, quantile=0.9, min_length=1):
    errors = reconstruction_errors_per_timestep(model, x)
    cutoff = float(np.quantile(errors, quantile))
    flagged = np.flatnonzero(errors >= cutoff)
    if flagged.size == 0:
        peak_idx = int(np.argmax(errors))
        return peak_idx, min(peak_idx + 1, x.shape[0]), errors

    runs = []
    start = int(flagged[0])
    prev = int(flagged[0])
    for idx in flagged[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev + 1))
        start = idx
        prev = idx
    runs.append((start, prev + 1))

    best_start, best_end = max(runs, key=lambda pair: (pair[1] - pair[0], np.max(errors[pair[0]:pair[1]])))
    if best_end - best_start < min_length:
        center = int(np.argmax(errors[best_start:best_end])) + best_start
        half = max(min_length // 2, 0)
        best_start = max(0, center - half)
        best_end = min(x.shape[0], best_start + min_length)
        best_start = max(0, best_end - min_length)
    return int(best_start), int(best_end), errors


class AnomalyRepairExplainer:
    def __init__(
        self,
        model,
        threshold,
        *,
        score_fn,
        normal_core=None,
        td=2,
        n_samples=25,
        interval_quantile=0.9,
        min_interval_length=2,
        enforce_psd=True,
        fallback_top_k=8,
        fallback_alpha_steps=9,
        context_width=2,
        crossfade_width=2,
        random_seed=42,
    ):
        self.model = model
        self.threshold = float(threshold)
        if not callable(score_fn):
            raise ValueError("anomaly_repair requires a callable score_fn")
        self.score_fn = score_fn
        self.normal_core = (
            None if normal_core is None else np.asarray(normal_core, dtype=np.float64)
        )
        self.td = int(td)
        self.n_samples = int(n_samples)
        self.interval_quantile = float(interval_quantile)
        self.min_interval_length = int(min_interval_length)
        self.enforce_psd = bool(enforce_psd)
        self.fallback_top_k = int(fallback_top_k)
        self.fallback_alpha_steps = int(fallback_alpha_steps)
        self.context_width = int(context_width)
        self.crossfade_width = int(crossfade_width)
        self.rng = np.random.default_rng(random_seed)
        self._current_interval = None

    def set_interval(self, interval):
        if interval is None:
            self._current_interval = None
            return
        if len(interval) != 2:
            raise ValueError("interval must be a (start, end) pair")

        start, end = int(interval[0]), int(interval[1])
        if start < 0 or end <= start:
            raise ValueError("interval must satisfy 0 <= start < end")
        self._current_interval = (start, end)

    def _score(self, x):
        return float(self.score_fn(np.asarray(x, dtype=np.float64)))

    def _context_distance(self, x, donor, start, end):
        left_start = max(0, start - self.context_width)
        left_end = start
        right_start = end
        right_end = min(x.shape[0], end + self.context_width)

        parts = []
        if left_end > left_start:
            parts.append(np.mean((x[left_start:left_end] - donor[left_start:left_end]) ** 2))
        if right_end > right_start:
            parts.append(np.mean((x[right_start:right_end] - donor[right_start:right_end]) ** 2))

        if parts:
            return float(np.mean(parts))
        return float(np.mean((x - donor) ** 2))

    def _try_donor_guided_repair(self, x_arr, score_before, start, end, interval_meta):
        if self.normal_core is None or self.normal_core.ndim != 3 or self.normal_core.shape[0] == 0:
            return None

        donor_indices = []
        donor_pool = []
        donor_dists = []
        for donor_idx, donor in enumerate(self.normal_core):
            if donor.shape != x_arr.shape:
                continue
            donor_indices.append(int(donor_idx))
            donor_pool.append(donor)
            donor_dists.append(self._context_distance(x_arr, donor, start, end))

        donor_pool = np.asarray(donor_pool, dtype=np.float64)
        donor_dists = np.asarray(donor_dists, dtype=np.float64)
        if donor_pool.shape[0] == 0 or donor_dists.size == 0:
            return None

        k = max(1, min(self.fallback_top_k, int(donor_pool.shape[0])))
        top_idx = np.argpartition(donor_dists, k - 1)[:k]
        top_idx = top_idx[np.argsort(donor_dists[top_idx])]
        alphas = np.linspace(1.0, 0.25, max(2, self.fallback_alpha_steps))
        shortlist_donors = [
            {
                "donor_idx": int(donor_indices[int(idx)]),
                "context_distance": float(donor_dists[int(idx)]),
            }
            for idx in top_idx[: min(5, len(top_idx))].tolist()
        ]

        best_candidate = None
        best_score = np.inf
        best_meta = None

        for donor_idx in top_idx:
            donor = donor_pool[int(donor_idx)]
            donor_slice = donor[start:end, :]
            for alpha in alphas:
                candidate = np.array(x_arr, copy=True)
                candidate[start:end, :] = (
                    (1.0 - float(alpha)) * x_arr[start:end, :]
                    + float(alpha) * donor_slice
                )
                candidate = _crossfade_boundaries(
                    candidate,
                    x_arr,
                    start=start,
                    end=end,
                    width=self.crossfade_width,
                )

                score = self._score(candidate)
                meta = {
                    "method": "anomaly_repair",
                    "repair_strategy": "donor_guided",
                    "interval": (int(start), int(end)),
                    "interval_source": "shared_model_localizer",
                    "score_before": float(score_before),
                    "score_source": "external_score_fn",
                    "shared_interval_quantile": float(self.interval_quantile),
                    "donor_idx": int(donor_indices[int(donor_idx)]),
                    "donor_context_distance": float(donor_dists[int(donor_idx)]),
                    "alpha": float(alpha),
                    "fallback_top_k": int(self.fallback_top_k),
                    "fallback_alpha_steps": int(self.fallback_alpha_steps),
                    "donor_shortlist": shortlist_donors,
                    "n_attempts": int(len(top_idx) * len(alphas)),
                }
                meta.update(build_score_summary(score_before, score, self.threshold))
                meta.update(interval_meta)
                if score < best_score:
                    best_candidate = candidate
                    best_score = score
                    best_meta = meta
                if score <= self.threshold:
                    return CFResult(x_cf=candidate, score_cf=float(score), meta=meta)

        return best_candidate, best_score, best_meta

    def explain(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 2:
            return CFFailure(
                reason="invalid_input",
                message="anomaly_repair expects x with shape (L, F)",
                diagnostics={"shape": tuple(x_arr.shape)},
            )

        score_before = self._score(x_arr)
        errors = reconstruction_errors_per_timestep(self.model, x_arr)
        if score_before <= self.threshold:
            return CFFailure(
                reason="already_valid",
                message="input is already below the anomaly threshold",
                diagnostics=build_score_summary(score_before, score_before, self.threshold),
            )

        if self._current_interval is None:
            return CFFailure(
                reason="missing_interval",
                message="anomaly_repair requires an externally supplied interval",
                diagnostics={},
            )

        start, end = self._current_interval
        if end > x_arr.shape[0]:
            return CFFailure(
                reason="invalid_interval",
                message="provided interval exceeds input window length",
                diagnostics={
                    "interval": (int(start), int(end)),
                    "window_length": int(x_arr.shape[0]),
                },
            )
        interval_meta = _interval_summary(start, end, x_arr.shape[0], errors)

        original_state = np.random.get_state()
        best_candidate = None
        best_score = np.inf
        best_meta = None
        gaussian_error = None
        gaussian_best_score = None
        try:
            for _ in range(self.n_samples):
                np.random.seed(int(self.rng.integers(0, 2**31 - 1)))
                replacement = sample_replacement(
                    x_arr.T,
                    (start, end),
                    td=self.td,
                    cond=None,
                    enforce_psd=self.enforce_psd,
                ).T
                candidate = x_arr.copy()
                candidate[start:end, :] = replacement
                score = self._score(candidate)
                if score < best_score:
                    best_candidate = candidate
                    best_score = score
                    gaussian_best_score = float(score)
                    best_meta = {
                        "method": "anomaly_repair",
                        "repair_strategy": "gaussian",
                        "interval_source": "shared_model_localizer",
                        "score_source": "external_score_fn",
                        "repair_samples": int(self.n_samples),
                        "shared_interval_quantile": float(self.interval_quantile),
                    }
                    best_meta.update(build_score_summary(score_before, score, self.threshold))
                    best_meta.update(interval_meta)
                if score <= self.threshold:
                    best_meta.update(build_explainability_meta(x_arr, candidate))
                    return CFResult(
                        x_cf=candidate,
                        score_cf=float(score),
                        meta=dict(best_meta),
                    )
        except Exception as exc:
            gaussian_error = str(exc)
        finally:
            np.random.set_state(original_state)

        donor_result = self._try_donor_guided_repair(
            x_arr,
            score_before,
            start,
            end,
            interval_meta,
        )
        donor_best_score = None
        if isinstance(donor_result, CFResult):
            donor_result.meta.update(build_explainability_meta(x_arr, donor_result.x_cf))
            return donor_result
        if donor_result is not None:
            donor_candidate, donor_score, donor_meta = donor_result
            donor_best_score = float(donor_score)
            if donor_score < best_score:
                best_candidate = donor_candidate
                best_score = donor_score
                best_meta = donor_meta

        if gaussian_error is not None and best_candidate is None:
            return CFFailure(
                reason="repair_failed",
                message=gaussian_error,
                diagnostics={
                    **build_score_summary(score_before, None, self.threshold),
                    **interval_meta,
                    "gaussian_error": gaussian_error,
                },
            )

        best_candidate_summary = None
        if best_candidate is not None and np.isfinite(best_score):
            best_candidate_summary = build_explainability_meta(x_arr, best_candidate)
            best_candidate_summary.update(
                build_score_summary(score_before, float(best_score), self.threshold)
            )
            if best_meta is not None:
                best_candidate_summary["repair_strategy"] = best_meta.get("repair_strategy")

        return CFFailure(
            reason="no_valid_cf",
            message="anomaly_repair did not find a valid counterfactual below the threshold",
            diagnostics={
                **build_score_summary(
                    score_before,
                    None if not np.isfinite(best_score) else float(best_score),
                    self.threshold,
                ),
                **interval_meta,
                "best_candidate_found": best_candidate is not None,
                "best_repair_strategy": None if best_meta is None else best_meta["repair_strategy"],
                "best_candidate_summary": best_candidate_summary,
                "gaussian_best_score": gaussian_best_score,
                "donor_guided_best_score": donor_best_score,
                "gaussian_error": gaussian_error,
            },
        )
