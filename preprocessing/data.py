import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from configs.defaults import (
    CALIB_RATIO,
    EXPECTED_FEATURE_COUNT,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    PRECIPITATION_LABEL_COLUMN,
    PRECIPITATION_LABEL_THRESHOLD,
    RESAMPLE_FREQUENCY,
    SCENARIO_COLUMN,
    SPLIT_MODE,
    TIME_COLUMN,
    TIME_UNIT,
    TRAIN_RATIO,
    TRAIN_SCENARIOS,
    VAL_RATIO,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)


def load_feature_names(path):
    if not path:
        return None
    with open(path, "r") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def _read_dataframe(data_path):
    suffix = os.path.splitext(data_path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(data_path)
    raise ValueError(f"Unsupported data format for {data_path!r}")


def _is_ignored_column(column_name):
    if column_name is None:
        return True
    text = str(column_name).strip()
    return (not text) or text.lower().startswith("unnamed:")


def _resolve_feature_names(df, feature_names):
    if feature_names:
        missing = [name for name in feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Configured feature columns missing from dataset: {missing}")
        return feature_names

    excluded = set(NON_FEATURE_COLUMNS)
    inferred = []
    for column in df.columns:
        if _is_ignored_column(column) or column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            inferred.append(column)

    if not inferred:
        raise ValueError("Unable to infer any numeric feature columns from the dataset")
    if EXPECTED_FEATURE_COUNT is not None and len(inferred) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns, found {len(inferred)}: {inferred}"
        )
    return inferred


def _parse_time_column(df):
    if TIME_COLUMN not in df.columns:
        return df
    if TIME_UNIT:
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], unit=TIME_UNIT)
    else:
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    return df


def _apply_missing_value_policy(df, feature_names):
    if SCENARIO_COLUMN and SCENARIO_COLUMN in df.columns:
        grouped = df.groupby(SCENARIO_COLUMN, sort=False)[feature_names]
        df[feature_names] = grouped.ffill(limit=3)
    else:
        df[feature_names] = df[feature_names].ffill(limit=3)
    df[feature_names] = df[feature_names].fillna(df[feature_names].median())
    return df


def _finalize_labels(df):
    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna(0).astype(int)
    else:
        df[LABEL_COLUMN] = 0

    if (
        PRECIPITATION_LABEL_COLUMN
        and PRECIPITATION_LABEL_COLUMN in df.columns
        and PRECIPITATION_LABEL_THRESHOLD is not None
    ):
        df.loc[df[PRECIPITATION_LABEL_COLUMN] > PRECIPITATION_LABEL_THRESHOLD, LABEL_COLUMN] = 1
    return df


def load_and_clean_dataframe(data_path, feature_names, output_dir):
    df = _read_dataframe(data_path)
    df = df.loc[:, [column for column in df.columns if not _is_ignored_column(column)]].copy()
    df = _parse_time_column(df)
    feature_names = _resolve_feature_names(df, feature_names)

    if SPLIT_MODE == "temporal":
        if TIME_COLUMN not in df.columns:
            raise ValueError(f"Temporal dataset must include time column {TIME_COLUMN!r}")
        df = df.sort_values(TIME_COLUMN)
        df = df.drop_duplicates(subset=[TIME_COLUMN], keep="first")
        df = df.set_index(TIME_COLUMN)
        if RESAMPLE_FREQUENCY:
            df = df.asfreq(RESAMPLE_FREQUENCY)
    else:
        sort_columns = []
        if SCENARIO_COLUMN and SCENARIO_COLUMN in df.columns:
            sort_columns.append(SCENARIO_COLUMN)
        if TIME_COLUMN in df.columns:
            sort_columns.append(TIME_COLUMN)
        if sort_columns:
            df = df.sort_values(sort_columns)
        if SCENARIO_COLUMN and TIME_COLUMN in df.columns:
            df = df.drop_duplicates(subset=[SCENARIO_COLUMN, TIME_COLUMN], keep="first")

    df = _apply_missing_value_policy(df, feature_names)
    df = _finalize_labels(df)

    final_columns = list(feature_names) + [LABEL_COLUMN]
    if SCENARIO_COLUMN and SCENARIO_COLUMN in df.columns:
        final_columns.append(SCENARIO_COLUMN)
    df_clean = df[final_columns].copy()
    df_clean.to_pickle(f"{output_dir}/01_ingested_qa_data.pkl")

    print(f"Cleaned Shape: {df_clean.shape}")
    print(f"Schema Locked. Features ({len(feature_names)}): {feature_names}")
    return df_clean, feature_names


def scale_feature_dataframe(df_clean, feature_names):
    df_scaled = df_clean.copy()
    df_features = df_scaled[feature_names]

    if SPLIT_MODE == "scenario":
        if not SCENARIO_COLUMN or SCENARIO_COLUMN not in df_scaled.columns:
            raise ValueError("Scenario split requested but scenario column is missing")
        train_mask = df_scaled[SCENARIO_COLUMN].isin(TRAIN_SCENARIOS)
        fit_frame = df_features.loc[train_mask]
        fit_description = f"training scenarios {tuple(TRAIN_SCENARIOS)}"
    else:
        train_cutoff = int(len(df_features) * TRAIN_RATIO)
        fit_frame = df_features.iloc[:train_cutoff]
        fit_description = f"first {train_cutoff} rows"

    if fit_frame.empty:
        raise ValueError("Scaler fit frame is empty; check split configuration")

    scaler = StandardScaler()
    scaler.fit(fit_frame)
    df_scaled.loc[:, feature_names] = scaler.transform(df_features)
    print(f"Data scaled. Scaler fit on {fit_description} strictly.")
    return df_scaled, scaler


def _window_array(array_2d):
    if int(WINDOW_STRIDE) <= 0:
        raise ValueError("WINDOW_STRIDE must be a positive integer")
    if len(array_2d) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, array_2d.shape[1]), dtype=np.float64)

    windows = np.lib.stride_tricks.sliding_window_view(
        array_2d,
        window_shape=(WINDOW_SIZE, array_2d.shape[1]),
    )
    windows = windows.squeeze(axis=1)
    return windows[:: int(WINDOW_STRIDE)]


def _window_labels(label_array):
    if len(label_array) < WINDOW_SIZE:
        return np.empty((0,), dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(label_array, window_shape=WINDOW_SIZE)
    return windows[:: int(WINDOW_STRIDE)].max(axis=1).astype(int)


def _concat_window_groups(groups, n_features):
    if groups:
        return np.concatenate(groups, axis=0)
    return np.empty((0, WINDOW_SIZE, n_features), dtype=np.float64)


def _concat_label_groups(groups):
    if groups:
        return np.concatenate(groups, axis=0)
    return np.empty((0,), dtype=int)


def build_windows(df_scaled, feature_names):
    if SPLIT_MODE == "scenario":
        train_groups = []
        train_label_groups = []
        test_groups = []
        test_label_groups = []

        for scenario_id, scenario_df in df_scaled.groupby(SCENARIO_COLUMN, sort=False):
            feature_array = scenario_df[feature_names].to_numpy(dtype=np.float64, copy=False)
            label_array = scenario_df[LABEL_COLUMN].to_numpy(dtype=int, copy=False)
            windows = _window_array(feature_array)
            window_labels = _window_labels(label_array)

            if scenario_id in TRAIN_SCENARIOS:
                train_groups.append(windows)
                train_label_groups.append(window_labels)
            else:
                test_groups.append(windows)
                test_label_groups.append(window_labels)

        payload = {
            "train_windows": _concat_window_groups(train_groups, len(feature_names)),
            "train_labels": _concat_label_groups(train_label_groups),
            "test_windows": _concat_window_groups(test_groups, len(feature_names)),
            "test_labels": _concat_label_groups(test_label_groups),
        }
        print(
            "Scenario windowing complete. "
            f"Train windows: {payload['train_windows'].shape}, "
            f"Test windows: {payload['test_windows'].shape}"
        )
        return payload

    windows = _window_array(df_scaled[feature_names].to_numpy(dtype=np.float64, copy=False))
    print(
        f"Windowing complete. Tensor shape: {windows.shape} "
        f"(window_size={WINDOW_SIZE}, stride={int(WINDOW_STRIDE)})"
    )
    return {"all_windows": windows}


def split_windows(window_payload):
    if SPLIT_MODE == "scenario":
        train_windows = window_payload["train_windows"]
        train_labels = window_payload["train_labels"]
        test_windows = window_payload["test_windows"]
        test_labels = window_payload["test_labels"]

        n_train_windows = len(train_windows)
        val_start = int(n_train_windows * TRAIN_RATIO)
        calib_start = val_start + int(n_train_windows * VAL_RATIO)

        splits = {
            "train": train_windows[:val_start],
            "val": train_windows[val_start:calib_start],
            "calib": train_windows[calib_start:],
            "test": test_windows,
            "labels_train": train_labels[:val_start],
            "labels_val": train_labels[val_start:calib_start],
            "labels_calib": train_labels[calib_start:],
            "labels_test": test_labels,
        }
    else:
        windows = window_payload["all_windows"]
        n_windows = len(windows)
        val_start = int(n_windows * TRAIN_RATIO)
        calib_start = val_start + int(n_windows * VAL_RATIO)
        test_start = calib_start + int(n_windows * CALIB_RATIO)

        splits = {
            "train": windows[:val_start],
            "val": windows[val_start:calib_start],
            "calib": windows[calib_start:test_start],
            "test": windows[test_start:],
        }

    print(f"X_train shape: {splits['train'].shape}")
    print(f"X_val shape:   {splits['val'].shape}")
    print(f"X_calib shape: {splits['calib'].shape}")
    print(f"X_test shape:  {splits['test'].shape}")
    return splits


def save_split_arrays(output_dir, splits):
    np.save(f"{output_dir}/X_train.npy", splits["train"])
    np.save(f"{output_dir}/X_val.npy", splits["val"])
    np.save(f"{output_dir}/X_calib.npy", splits["calib"])
    np.save(f"{output_dir}/X_test.npy", splits["test"])

    for name in ("labels_train", "labels_val", "labels_calib", "labels_test"):
        if name in splits:
            np.save(f"{output_dir}/{name}.npy", splits[name])
