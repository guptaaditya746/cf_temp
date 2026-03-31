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


def _extract_best_failure_score(result):
    if not isinstance(result, CFFailure):
        return None

    diagnostics = result.diagnostics or {}
    for key in ("score_after", "gaussian_best_score", "donor_guided_best_score"):
        value = diagnostics.get(key)
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def _relaxed_retry_overrides(explainer):
    if isinstance(explainer, AnomalyRepairExplainer):
        overrides = {
            "n_samples": max(int(getattr(explainer, "n_samples", 1)), 50),
            "fallback_top_k": max(int(getattr(explainer, "fallback_top_k", 1)), 12),
            "fallback_alpha_steps": max(int(getattr(explainer, "fallback_alpha_steps", 2)), 15),
        }
        if hasattr(explainer, "max_features_to_try"):
            overrides["max_features_to_try"] = max(
                int(getattr(explainer, "max_features_to_try", 1)),
                12,
            )
        if hasattr(explainer, "subset_search_width"):
            overrides["subset_search_width"] = max(
                int(getattr(explainer, "subset_search_width", 1)),
                6,
            )
        if hasattr(explainer, "max_subset_size"):
            overrides["max_subset_size"] = max(
                int(getattr(explainer, "max_subset_size", 1)),
                3,
            )
        if hasattr(explainer, "subset_eval_samples"):
            overrides["subset_eval_samples"] = max(
                int(getattr(explainer, "subset_eval_samples", 1)),
                5,
            )
        return overrides

    method = getattr(explainer, "method", None)
    if method == "segment":
        return {
            "segment_n_candidates": max(int(getattr(explainer, "segment_n_candidates", 1)), 8),
            "segment_top_k_donors": max(int(getattr(explainer, "segment_top_k_donors", 1)), 16),
            "segment_max_pair_groups": max(int(getattr(explainer, "segment_max_pair_groups", 0)), 8),
            "segment_context_width": max(int(getattr(explainer, "segment_context_width", 0)), 3),
        }
    if method == "motif":
        return {
            "motif_n_segments": max(int(getattr(explainer, "motif_n_segments", 1)), 8),
            "motif_top_k": max(int(getattr(explainer, "motif_top_k", 1)), 12),
            "motif_max_pair_groups": max(int(getattr(explainer, "motif_max_pair_groups", 0)), 8),
        }
    if method == "nearest":
        return {
            "nearest_top_k": max(int(getattr(explainer, "nearest_top_k", 1)), 20),
            "nearest_alpha_steps": max(int(getattr(explainer, "nearest_alpha_steps", 2)), 21),
        }
    return {}


def _run_with_overrides(explainer, x_anomaly, overrides):
    if not overrides:
        return explainer.explain(x_anomaly), {}

    original = {}
    for key, value in overrides.items():
        if hasattr(explainer, key):
            original[key] = getattr(explainer, key)
            setattr(explainer, key, value)

    try:
        return explainer.explain(x_anomaly), original
    finally:
        for key, value in original.items():
            setattr(explainer, key, value)


def _maybe_retry_with_relaxed_search(explainer, x_anomaly, result):
    if not isinstance(result, CFFailure) or result.reason != "no_valid_cf":
        return result

    overrides = _relaxed_retry_overrides(explainer)
    retry_result, applied = _run_with_overrides(explainer, x_anomaly, overrides)
    if not applied:
        return result

    if isinstance(retry_result, CFResult):
        retry_result.meta["relaxed_retry_used"] = True
        retry_result.meta["relaxed_retry_overrides"] = dict(applied)
        retry_result.meta["relaxed_retry_trigger"] = result.reason
        return retry_result

    original_best = _extract_best_failure_score(result)
    retry_best = _extract_best_failure_score(retry_result)
    if retry_best is not None and (original_best is None or retry_best < original_best):
        retry_result.diagnostics["relaxed_retry_used"] = True
        retry_result.diagnostics["relaxed_retry_overrides"] = dict(applied)
        retry_result.diagnostics["relaxed_retry_trigger"] = result.reason
        return retry_result

    result.diagnostics["relaxed_retry_used"] = True
    result.diagnostics["relaxed_retry_overrides"] = dict(applied)
    result.diagnostics["relaxed_retry_trigger"] = result.reason
    return result


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
        normal_core=normal_core,
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
        result = _maybe_retry_with_relaxed_search(explainer, x_anomaly, result)
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
            best_score = result.diagnostics.get("score_after")
            if best_score is None:
                best_score = result.diagnostics.get("gaussian_best_score")
            if best_score is None:
                best_score = result.diagnostics.get("donor_guided_best_score")

            if best_score is not None and np.isfinite(best_score):
                print(f"[{method_name}] no valid cf -> best_cf_score={float(best_score):.4f}")
                row["cf_score"] = float(best_score)
            else:
                print(f"[{method_name}] no valid cf -> best_cf_score=N/A")
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
            shared_start, shared_end, _, _, _ = detect_anomalous_interval(
                x_anomaly,
                model,
                normal_core=getattr(anomaly_repair, "normal_core", None),
                quantile=float(getattr(anomaly_repair, "interval_quantile", 0.9)),
                min_length=int(getattr(anomaly_repair, "min_interval_length", 1)),
                normalize_errors=bool(getattr(anomaly_repair, "normalize_errors", True)),
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
