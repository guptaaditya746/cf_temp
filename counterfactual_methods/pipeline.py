import csv
import json
import os

import numpy as np
import pandas as pd

from configs.defaults import ANOMALY_REPAIR_CONFIG, CFTSAD_BASE_CONFIG, CFTSAD_METHOD_CONFIGS
from counterfactual_methods.anomaly_repair import (
    AnomalyRepairExplainer,
    detect_anomalous_interval,
)
from evaluation.reconstruction import build_score_fn

from cftsad import CFFailure, CFResult, CounterfactualExplainer




def _safe_log_value(value):
    if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, dict):
        value = {
            str(key): _safe_log_value(item)
            for key, item in value.items()
        }
    elif isinstance(value, list):
        value = [_safe_log_value(item) for item in value]

    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _append_counterfactual_log(csv_path, df_log):
    write_kwargs = {"index": False, "quoting": csv.QUOTE_ALL}
    if not os.path.exists(csv_path):
        df_log.to_csv(csv_path, **write_kwargs)
        return

    existing_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    new_columns = df_log.columns.tolist()
    extra_columns = [column for column in new_columns if column not in existing_columns]

    if not extra_columns:
        df_log.reindex(columns=existing_columns).to_csv(
            csv_path,
            mode="a",
            header=False,
            **write_kwargs,
        )
        return

    existing_df = pd.read_csv(csv_path)
    all_columns = existing_columns + extra_columns
    combined_df = pd.concat(
        [
            existing_df.reindex(columns=all_columns),
            df_log.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    combined_df.to_csv(csv_path, **write_kwargs)


def build_cftsad_explainers(model, normal_core, threshold, score_fn):
    base_kwargs = {
        "model": model,
        "normal_core": normal_core,
        "score_fn": score_fn,
        "threshold": float(threshold),
        **CFTSAD_BASE_CONFIG,
    }

    explainers = {}
    for method, overrides in CFTSAD_METHOD_CONFIGS.items():
        explainers[method] = CounterfactualExplainer(
            method=method,
            **base_kwargs,
            **overrides,
        )
    explainers["anomaly_repair"] = AnomalyRepairExplainer(
        model=model,
        threshold=float(threshold),
        score_fn=score_fn,
        **ANOMALY_REPAIR_CONFIG,
    )
    return explainers


def run_counterfactual_benchmark(
    explainers,
    x_anomaly,
    idx_to_explain,
    original_score,
    threshold,
    eval_dir,
):
    csv_path = os.path.join(eval_dir, "counterfactual_log.csv")
    cf_arrays_dir = os.path.join(eval_dir, "cf_arrays")
    os.makedirs(cf_arrays_dir, exist_ok=True)

    rows = []
    results_by_method = {}
    for method_name, explainer in explainers.items():
        result = explainer.explain(x_anomaly)
        results_by_method[method_name] = result

        row = {
            "method": method_name,
            "test_index": int(idx_to_explain),
            "original_score": float(original_score),
            "target_threshold": float(threshold),
            "status": "success" if isinstance(result, CFResult) else "failed",
        }

        if isinstance(result, CFResult):
            print(f"[{method_name}] success -> cf_score={result.score_cf:.4f}")
            row["cf_score"] = float(result.score_cf)
            row["reason"] = "N/A"
            row["message"] = "N/A"
            cf_filename = f"cf_{method_name}_window_{idx_to_explain}.npy"
            np.save(os.path.join(cf_arrays_dir, cf_filename), result.x_cf)
            row["cf_array_file"] = cf_filename
            for key, value in result.meta.items():
                row[f"meta_{key}"] = _safe_log_value(value)
        else:
            print(f"[{method_name}] failed -> {result.reason}")
            row["cf_score"] = np.nan
            row["reason"] = result.reason
            row["message"] = result.message
            row["cf_array_file"] = "N/A"
            for key, value in result.diagnostics.items():
                row[f"diag_{key}"] = _safe_log_value(value)

        rows.append(row)

    df_log = pd.DataFrame(rows)
    _append_counterfactual_log(csv_path, df_log)

    print(f"Counterfactual results appended to: {csv_path}")
    return results_by_method


def run_counterfactual_pipeline(model, splits, evaluation, eval_dir):
    score_fn = build_score_fn(model)
    anomaly_indices = np.where(
        evaluation["test_window_mse"] > evaluation["window_threshold"]
    )[0]
    if len(anomaly_indices) == 0:
        raise RuntimeError("No anomalies found in test set for counterfactual generation.")

    explainers = build_cftsad_explainers(
        model=model,
        normal_core=splits["calib"],
        threshold=evaluation["window_threshold"],
        score_fn=score_fn,
    )

    print(f"Found {len(anomaly_indices)} anomaly windows in test set.")
    print(f"Target threshold: {evaluation['window_threshold']:.4f}")
    print("Explainers initialized:", ", ".join(explainers.keys()))

    anomaly_repair = explainers.get("anomaly_repair")
    results_by_index = {}
    for idx in anomaly_indices:
        idx_to_explain = int(idx)
        x_anomaly = splits["test"][idx_to_explain]
        original_score = float(evaluation["test_window_mse"][idx_to_explain])

        print(f"\nGenerating counterfactuals for Test Window Index: {idx_to_explain}")
        print(f"Original score: {original_score:.4f}")

        if anomaly_repair is not None:
            shared_start, shared_end, _ = detect_anomalous_interval(
                x_anomaly,
                model,
                quantile=float(getattr(anomaly_repair, "interval_quantile", 0.9)),
                min_length=int(getattr(anomaly_repair, "min_interval_length", 1)),
            )
            anomaly_repair.set_interval((shared_start, shared_end))
            print(f"Shared interval for this window: [{shared_start}, {shared_end})")

        results_by_method = run_counterfactual_benchmark(
            explainers=explainers,
            x_anomaly=x_anomaly,
            idx_to_explain=idx_to_explain,
            original_score=original_score,
            threshold=evaluation["window_threshold"],
            eval_dir=eval_dir,
        )
        results_by_index[idx_to_explain] = results_by_method

    return results_by_index
