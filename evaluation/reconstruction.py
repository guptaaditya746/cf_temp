import os

import numpy as np
import torch

from configs.defaults import THRESHOLD_PERCENTILE


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def predict_reconstruction(model, x):
    x_arr = np.asarray(x, dtype=np.float32)
    expected_shape = x_arr.shape

    if isinstance(model, torch.nn.Module):
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
        _, window_mse = calculate_errors(x_arr[np.newaxis, ...], recon[np.newaxis, ...])
        return float(window_mse[0])

    return score_fn


def run_model_inference(model, dataloader, device):
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_reconstruction_model(model, data_module, splits, device):
    x_calib_hat = run_model_inference(model, data_module.predict_dataloader(), device)
    x_test_hat = run_model_inference(model, data_module.test_dataloader(), device)

    calib_feature_mse, calib_window_mse = calculate_errors(splits["calib"], x_calib_hat)
    test_feature_mse, test_window_mse = calculate_errors(splits["test"], x_test_hat)

    window_threshold = np.percentile(calib_window_mse, THRESHOLD_PERCENTILE)
    feature_thresholds = np.percentile(
        calib_feature_mse,
        THRESHOLD_PERCENTILE,
        axis=0,
    )

    print(f"Calibration Arrays Shape: {splits['calib'].shape}")
    print(f"Test Arrays Shape: {splits['test'].shape}")
    print(
        f"Global Window Threshold ({THRESHOLD_PERCENTILE}th percentile): "
        f"{window_threshold:.4f}"
    )

    return {
        "x_calib_hat": x_calib_hat,
        "x_test_hat": x_test_hat,
        "calib_feature_mse": calib_feature_mse,
        "calib_window_mse": calib_window_mse,
        "test_feature_mse": test_feature_mse,
        "test_window_mse": test_window_mse,
        "window_threshold": window_threshold,
        "feature_thresholds": feature_thresholds,
    }


def save_evaluation_outputs(output_dir, evaluation, feature_names):
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    test_window_anomalies = (
        evaluation["test_window_mse"] > evaluation["window_threshold"]
    ).astype(int)
    test_feature_anomalies = (
        evaluation["test_feature_mse"] > evaluation["feature_thresholds"]
    ).astype(int)

    np.save(os.path.join(eval_dir, "window_threshold.npy"), evaluation["window_threshold"])
    np.save(os.path.join(eval_dir, "feature_thresholds.npy"), evaluation["feature_thresholds"])
    np.save(os.path.join(eval_dir, "test_window_mse.npy"), evaluation["test_window_mse"])
    np.save(os.path.join(eval_dir, "test_feature_mse.npy"), evaluation["test_feature_mse"])
    np.save(os.path.join(eval_dir, "test_window_anomalies.npy"), test_window_anomalies)
    np.save(os.path.join(eval_dir, "test_feature_anomalies.npy"), test_feature_anomalies)

    print("Feature-Level Thresholds:")
    for feat, thresh in zip(feature_names, evaluation["feature_thresholds"]):
        print(f" - {feat}: {thresh:.4f}")
    print(f"Total Window Anomalies Found in Test Set: {test_window_anomalies.sum()}")
    print(f"All evaluation outputs and anomaly scores successfully stored in: {eval_dir}")

    return eval_dir, test_window_anomalies
