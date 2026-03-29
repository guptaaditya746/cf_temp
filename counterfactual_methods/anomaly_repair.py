import numpy as np
from scipy.stats import multivariate_normal as mvn

try:
    from cftsad.core.scoring import reconstruction_errors_per_timestep, reconstruction_score
    from cftsad.types import CFFailure, CFResult
except Exception:
    from src.cftsad.core.scoring import reconstruction_errors_per_timestep, reconstruction_score
    from src.cftsad.types import CFFailure, CFResult


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
        eigvals, eigvec = np.linalg.eigh(cov)
        if np.any(eigvals < 0):
            cov = (eigvec * np.maximum(0, eigvals[None, :])) @ eigvec.T

    # Condition on observed variables and left and right context
    if td > 1:
        context_conditions = np.concatenate([np.arange((td - 1) * dim), np.arange((length + td - 1) * dim, len(rep_mean))])
        variable_conditions = np.concatenate([np.arange(d, len(rep_mean), dim) for d in cond]) \
                              if cond is not None and len(cond) > 0 \
                              else np.zeros((0,), dtype=int)

        rep_mean, cov = conditional_mvn(
            rep_mean, cov,
            X[:, intvl[0]-td+1:intvl[1]+td-1].T.ravel(),
            np.union1d(context_conditions, variable_conditions)
        )

    # Sample replacement
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

    # Partition covariance matrix
    S11 = S[np.ix_(d_inf, d_inf)]
    S12 = S[np.ix_(d_inf, d_obs)]
    S22 = S[np.ix_(d_obs, d_obs)]

    # Compute conditional mean and covariance
    if X.ndim > 1 and mu.ndim == 1:
        mu = mu[:, None]
    mu_cond = mu[d_inf, ...] + S12 @ np.linalg.solve(S22, X[d_obs, ...] - mu[d_obs, ...])
    S_cond = S11 - S12 @ np.linalg.solve(S22, S12.T)
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
        score_fn=None,
        td=2,
        n_samples=25,
        interval_quantile=0.9,
        min_interval_length=2,
        enforce_psd=True,
        random_seed=42,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.score_fn = score_fn
        self.td = int(td)
        self.n_samples = int(n_samples)
        self.interval_quantile = float(interval_quantile)
        self.min_interval_length = int(min_interval_length)
        self.enforce_psd = bool(enforce_psd)
        self.rng = np.random.default_rng(random_seed)

    def _score(self, x):
        if self.score_fn is not None:
            return float(self.score_fn(np.asarray(x, dtype=np.float64)))
        return reconstruction_score(self.model, x)

    def explain(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 2:
            return CFFailure(
                reason="invalid_input",
                message="anomaly_repair expects x with shape (L, F)",
                diagnostics={"shape": tuple(x_arr.shape)},
            )

        score_before = self._score(x_arr)
        if score_before <= self.threshold:
            return CFFailure(
                reason="already_valid",
                message="input is already below the anomaly threshold",
                diagnostics={"score_before": float(score_before)},
            )

        start, end, errors = detect_anomalous_interval(
            x_arr,
            self.model,
            quantile=self.interval_quantile,
            min_length=self.min_interval_length,
        )

        original_state = np.random.get_state()
        best_candidate = None
        best_score = np.inf
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
                if score <= self.threshold:
                    meta = {
                        "method": "anomaly_repair",
                        "interval": (int(start), int(end)),
                        "score_before": float(score_before),
                        "score_source": "custom" if self.score_fn is not None else "reconstruction_score",
                        "repair_samples": int(self.n_samples),
                        "interval_quantile": float(self.interval_quantile),
                        "peak_error": float(np.max(errors)),
                    }
                    return CFResult(x_cf=candidate, score_cf=float(score), meta=meta)
        except Exception as exc:
            return CFFailure(
                reason="repair_failed",
                message=str(exc),
                diagnostics={"interval": (int(start), int(end))},
            )
        finally:
            np.random.set_state(original_state)

        return CFFailure(
            reason="no_valid_cf",
            message="anomaly_repair did not find a valid counterfactual below the threshold",
            diagnostics={
                "score_before": float(score_before),
                "best_score": float(best_score) if np.isfinite(best_score) else None,
                "interval": (int(start), int(end)),
                "best_candidate_found": best_candidate is not None,
            },
        )
