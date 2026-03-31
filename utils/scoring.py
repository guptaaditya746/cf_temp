import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def predict_reconstruction(model, x):
    x_arr = np.asarray(x, dtype=np.float32)
    expected_shape = x_arr.shape

    if torch is not None and isinstance(model, torch.nn.Module):
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        model.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=device)
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0)
            out = _as_numpy(model(x_tensor))
    else:
        out = _as_numpy(model(x_arr))
        if out.shape != expected_shape and x_arr.ndim == 2:
            out = _as_numpy(model(x_arr[np.newaxis, ...]))

    if out.ndim == 3 and out.shape[0] == 1 and out.shape[1:] == expected_shape:
        out = out[0]
    if out.shape != expected_shape:
        raise ValueError(
            f"Model reconstruction shape mismatch: expected {expected_shape}, got {out.shape}"
        )
    return out.astype(np.float64, copy=False)


def calculate_errors(true_data, pred_data):
    squared_error = np.square(true_data - pred_data)
    feature_mse = np.mean(squared_error, axis=1)
    window_mse = np.mean(squared_error, axis=(1, 2))
    return feature_mse, window_mse


def build_score_fn(model):
    def score_fn(x):
        x_arr = np.asarray(x, dtype=np.float64)
        recon = predict_reconstruction(model, x_arr)
        _, window_mse = calculate_errors(
            x_arr[np.newaxis, ...],
            recon[np.newaxis, ...],
        )
        return float(window_mse[0])

    return score_fn
