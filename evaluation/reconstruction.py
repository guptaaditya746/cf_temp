import os

import numpy as np
import torch

from configs.defaults import THRESHOLD_PERCENTILE


def run_model_inference(model, dataloader, device):
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def calculate_errors(true_data, pred_data):
    squared_error = np.square(true_data - pred_data)
    feature_mse = np.mean(squared_error, axis=1)
    window_mse = np.mean(squared_error, axis=(1, 2))
    return feature_mse, window_mse


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
