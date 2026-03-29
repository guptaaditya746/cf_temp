import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from configs.defaults import CALIB_RATIO, TRAIN_RATIO, VAL_RATIO, WINDOW_SIZE


def load_feature_names(path):
    with open(path, "r") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def load_and_clean_dataframe(data_path, feature_names, output_dir):
    df = pd.read_pickle(data_path)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("h")

    df[feature_names] = df[feature_names].ffill(limit=3)
    df[feature_names] = df[feature_names].fillna(df[feature_names].median())

    df["label"] = df["label"].fillna(0).astype(int)
    df.loc[df["precipitation (mm)"] > 2.0, "label"] = 1

    final_columns = feature_names + ["label"]
    df_clean = df[final_columns]
    df_clean.to_pickle(f"{output_dir}/01_ingested_qa_data.pkl")

    print(f"Cleaned Shape: {df_clean.shape}")
    print(f"Schema Locked. Features: {feature_names}")
    return df_clean


def scale_feature_dataframe(df_clean):
    df_features = df_clean.drop(columns=["label"])
    train_cutoff = int(len(df_features) * TRAIN_RATIO)

    scaler = StandardScaler()
    scaler.fit(df_features.iloc[:train_cutoff])

    scaled_data = scaler.transform(df_features)
    print(f"Data scaled. Scaler fit on first {train_cutoff} rows strictly.")
    return scaled_data, scaler


def build_windows(scaled_data):
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled_data,
        window_shape=(WINDOW_SIZE, scaled_data.shape[1]),
    )
    windows = windows.squeeze(axis=1)
    print(f"Windowing complete. Tensor shape: {windows.shape}")
    return windows


def split_windows(windows):
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
