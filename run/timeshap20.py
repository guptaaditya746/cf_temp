import copy
import json
import os
from dataclasses import dataclass
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Full TimeSHAP Functionality Imports
from git.timeshap.src.timeshap.explainer import (
    local_cell_level,
    local_event,
    local_feat,
    local_pruning,
    local_report,
)
from git.timeshap.src.timeshap.plot import (
    plot_cell_level,
    plot_event_heatmap,
    plot_feat_barplot,
    plot_temp_coalition_pruning,
)


# ============================================================================
# 1. TORCHSCRIPT PROXY (Ensures correct data types for scripted model)
# ============================================================================
class TorchScriptProxy:
    def __init__(
        self, model, threshold, device, num_layers=2, input_size=6, hidden_size=128
    ):
        self.model = model
        self.optimal_thresh = torch.tensor(threshold, device=device)
        self.device = device

        @dataclass
        class MockConfig:
            num_layers: int
            input_size: int
            hidden_size: int

        self.cfg = MockConfig(num_layers, input_size, hidden_size)

    @torch.no_grad()
    def score(self, x: Any) -> float:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device).to(torch.float32)
        x_hat = self.model(x)
        return float(torch.mean((x - x_hat) ** 2).item())


# ============================================================================
# 2. TIMESHAP WRAPPER
# ============================================================================
def create_timeshap_wrapper(proxy, threshold):
    def predict_for_timeshap(x):
        # TimeSHAP passes windows as numpy arrays; we return normal-probability
        scores = np.array([proxy.score(window) for window in x])
        probs_normal = 1.0 / (1.0 + np.exp(scores - threshold))
        return probs_normal.reshape(-1, 1)

    return predict_for_timeshap


# ============================================================================
# 3. MAIN EXECUTION (Batch Top 20)
# ============================================================================
def main():
    # --- CONFIGURATION ---
    OUTPUT_DIR = "/home/gupt_ad/conclusion_work/application/cf_final/run/experiments/benchmark_all_methods"
    DATA_PATH = (
        "/home/gupt_ad/conclusion_work/application/ad-xai_pipeline/ATACAMA/Atacama.pkl"
    )
    MASTER_EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "timeshap_top_20")
    os.makedirs(MASTER_EXPLAIN_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURE_COLUMNS = [
        "temperature_2m (°C)",
        "relative_humidity_2m (%)",
        "precipitation (mm)",
        "pressure_msl (hPa)",
        "cloud_cover (%)",
        "wind_speed_10m (km/h)",
    ]

    # --- 1. Load Data ---
    print("Loading data...")
    df = pd.read_pickle(DATA_PATH)
    df["label"] = df["label"].map({0: 0, 1: 1, 2: 1}).fillna(0)
    data_raw = df[FEATURE_COLUMNS].values
    data_norm = (data_raw - data_raw.mean(axis=0)) / (data_raw.std(axis=0) + 1e-8)

    X_windows = np.lib.stride_tricks.sliding_window_view(data_norm, 512, axis=0)[::1]
    X_windows = np.moveaxis(X_windows, -2, -1)
    labels = (
        np.lib.stride_tricks.sliding_window_view(df["label"].values, 512, axis=0).sum(
            axis=1
        )
        > 0
    ).astype(int)

    # --- 2. Load Model & Threshold ---
    with open(os.path.join(OUTPUT_DIR, "performance_summary.json"), "r") as f:
        perf = json.load(f)
        threshold = float(perf["optimal_threshold"])

    loaded_model = torch.jit.load(
        os.path.join(OUTPUT_DIR, "model_ts.pt"), map_location=DEVICE
    )
    proxy = TorchScriptProxy(loaded_model, threshold, DEVICE)
    f_timeshap = create_timeshap_wrapper(proxy, threshold)

    # --- 3. Identify Top 20 Anomalies ---
    print("Ranking anomalies by score...")
    anomaly_indices = np.where(labels == 1)[0]
    scores = [proxy.score(X_windows[idx]) for idx in anomaly_indices]
    top_20_indices = [
        idx for _, idx in sorted(zip(scores, anomaly_indices), reverse=True)[:20]
    ]

    # Baseline for TimeSHAP (Mean of normal data)
    baseline = X_windows[labels == 0][:100].mean(axis=0, keepdims=True)

    # --- 4. Explainer Configuration Dictionaries ---
    pruning_dict_orig = {"tol": 0.05, "rs": 42}
    event_dict_orig = {"rs": 42, "nsamples": 200}
    feature_dict_orig = {"rs": 42, "nsamples": 200, "feature_names": FEATURE_COLUMNS}
    cell_dict_orig = {"rs": 42, "nsamples": 200, "top_x_events": 5, "top_x_feats": 3}

    # --- 5. BATCH EXPLAIN LOOP ---
    # ... [Previous code: Loading Proxy, Model, and Ranking] ...

    # --- Batch Process TimeSHAP ---
    # Calculate a baseline from normal windows (crucial for SHAP)
    baseline = X_windows[labels == 0][:100].mean(axis=0, keepdims=True)

    for rank, idx in enumerate(top_20_indices):
        print(f"\n🚀 [{rank + 1}/20] Explaining Anomaly Index: {idx}")

        # 1. Create a unique directory for THIS specific anomaly
        current_save_dir = os.path.join(MASTER_EXPLAIN_DIR, f"top_{rank + 1}_idx_{idx}")
        os.makedirs(current_save_dir, exist_ok=True)

        instance = X_windows[idx : idx + 1]  # (1, 512, 6)

        # 2. Run TimeSHAP computations
        print(f"   Running Pruning and Local Explanations...")
        coal_data, prun_idx = local_pruning(
            f_timeshap, instance, {"tol": 0.05, "rs": 42}, baseline
        )
        effective_len = 512 + prun_idx

        event_exp = local_event(
            f_timeshap,
            instance,
            {"rs": 42, "nsamples": 200},
            None,
            None,
            baseline,
            effective_len,
        )
        feat_exp = local_feat(
            f_timeshap,
            instance,
            {"rs": 42, "nsamples": 200, "feature_names": FEATURE_COLUMNS},
            None,
            None,
            baseline,
            effective_len,
        )

        # 3. GENERATE PLOTS (Must be inside this loop to get 20 different ones)

        # Plot A: Feature Importance
        plt.figure(figsize=(10, 6))
        feat_data = feat_exp[feat_exp["Feature"] != "Pruned Events"].sort_values(
            "Shapley Value", ascending=True
        )
        colors = ["red" if x < 0 else "blue" for x in feat_data["Shapley Value"]]
        plt.barh(feat_data["Feature"], feat_data["Shapley Value"], color=colors)
        plt.title(
            f"Feature Importance: Top {rank + 1} (Idx {idx})\nScore: {scores[rank]:.4f}"
        )
        plt.grid(axis="x", alpha=0.3)
        plt.savefig(
            os.path.join(current_save_dir, "1_feature_importance.png"),
            bbox_inches="tight",
        )
        plt.close()  # Free memory

        # Plot B: Temporal Event Importance
        plt.figure(figsize=(12, 4))
        event_vals = event_exp[~event_exp["Feature"].str.contains("Pruned")][
            "Shapley Value"
        ].values
        plt.plot(event_vals, color="steelblue", lw=2)
        plt.fill_between(range(len(event_vals)), 0, event_vals, alpha=0.2)
        plt.axhline(0, color="black", lw=1)
        plt.title(f"Temporal Contribution: Top {rank + 1} (Idx {idx})")
        plt.xlabel("Relative Timestep")
        plt.ylabel("Shapley Value")
        plt.savefig(
            os.path.join(current_save_dir, "2_event_temporal.png"), bbox_inches="tight"
        )
        plt.close()

        # 4. Save CSV data for this anomaly
        event_exp.to_csv(os.path.join(current_save_dir, "event_data.csv"))
        feat_exp.to_csv(os.path.join(current_save_dir, "feature_data.csv"))

    print(f"\n✅ Done! Check the folder: {MASTER_EXPLAIN_DIR}")
    print("You should see 20 subfolders, each with their own unique plots.")

    print("You should see 20 subfolders, each with their own unique plots.")


if __name__ == "__main__":
    main()
