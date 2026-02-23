from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None


def _as_numpy(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_batch_if_needed(out: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    if out.shape == expected_shape:
        return out
    if (
        len(expected_shape) == 2
        and out.ndim == 3
        and out.shape[0] == 1
        and out.shape[1:] == expected_shape
    ):
        return out[0]
    return out


def _call_model(model: object, x: np.ndarray) -> np.ndarray:
    """Call model with numpy input and return reconstruction matching x shape."""
    expected_shape = x.shape

    if torch is not None and isinstance(model, torch.nn.Module):
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
            if x_t.ndim == 2:
                out_t = model(x_t.unsqueeze(0))
            else:
                out_t = model(x_t)
        out = _as_numpy(out_t)
        out = _squeeze_batch_if_needed(out, expected_shape)
    else:
        out = _as_numpy(model(x))
        out = _squeeze_batch_if_needed(out, expected_shape)
        if out.shape != expected_shape and x.ndim == 2:
            # Fallback for callables that require a batch dimension.
            out = _as_numpy(model(x[np.newaxis, ...]))
            out = _squeeze_batch_if_needed(out, expected_shape)

    if out.shape != expected_shape:
        raise ValueError(
            f"Model reconstruction shape mismatch: expected {expected_shape}, got {out.shape}"
        )
    return out


def per_timestep_mse(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Per-timestep MSE e_t = mean_f((x_t,f - x_hat_t,f)^2)."""
    return np.mean((x - x_hat) ** 2, axis=1)


def window_mse_score(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Window score = mean_t(e_t)."""
    return float(np.mean(per_timestep_mse(x, x_hat)))


def reconstruction_score(model: object, x: np.ndarray) -> float:
    x_hat = _call_model(model, x)
    return window_mse_score(x, x_hat)


def reconstruction_errors_per_timestep(model: object, x: np.ndarray) -> np.ndarray:
    x_hat = _call_model(model, x)
    return per_timestep_mse(x, x_hat)


def compute_threshold_from_normal_core(
    model: object,
    normal_core: np.ndarray,
    quantile: float = 0.95,
) -> float:
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")

    scores = [
        reconstruction_score(model, normal_core[i])
        for i in range(normal_core.shape[0])
    ]
    return float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))
