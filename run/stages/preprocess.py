from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.io import load_npz, save_npz, save_yaml
from ..utils.windowing import (
    WindowSpec,
    apply_scaling,
    compute_scaler_params,
    make_windows,
    split_windows,
)


@dataclass(frozen=True)
class PreprocessOut:
    train_path: Path
    calibration_path: Path
    validation_path: Path
    test_path: Path
    scaler_path: Path
    config_path: Path


def _load_raw_dataset(dataset: str, cfg_data: Dict[str, Any]) -> np.ndarray:
    """
    Replace this with your dataset adapters.
    Must return float32 array shaped (T,F).
    """
    raw_path = cfg_data.get("raw_path")
    if raw_path is None:
        raise ValueError("data.raw_path must be set (or implement a dataset resolver).")

    raw = load_npz(raw_path)
    # Expect key "x" as (T,F) OR (N,L,F) already. If already windowed, we'll treat it as windows.
    if "x" not in raw:
        raise ValueError(
            f"Raw NPZ must contain key 'x'. Found keys: {list(raw.keys())}"
        )

    x = raw["x"]
    if x.ndim == 2:
        return x.astype(np.float32, copy=False)
    if x.ndim == 3:
        # already windowed: (N,L,F) -> we will bypass make_windows and splitting expects windows
        return x.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported raw x shape: {x.shape}")


def run_preprocess(run_dir: Path, dataset: str, cfg: Dict[str, Any]) -> PreprocessOut:
    art = run_dir / "artifacts"

    cfg_data = cfg["data"]
    win_cfg = cfg_data["window"]
    spec = WindowSpec(length=int(win_cfg["length"]), stride=int(win_cfg["stride"]))

    x = _load_raw_dataset(dataset, cfg_data)

    if x.ndim == 2:
        w = make_windows(x, spec)  # (N,L,F)
    else:
        w = x  # already windowed

    splits = split_windows(w, cfg_data["splits"], seed=int(cfg["seed"]["seed"]))

    mean, std = compute_scaler_params(
        splits["train"], eps=float(cfg_data["scaler"]["eps"])
    )
    for k in list(splits.keys()):
        splits[k] = apply_scaling(splits[k], mean, std)

    # Save artifacts
    train_path = art / "train.npz"
    calibration_path = art / "calibration.npz"
    validation_path = art / "validation.npz"
    test_path = art / "test.npz"
    scaler_path = art / "scaler_params.npz"
    config_path = art / "config_data.yaml"

    save_npz(train_path, x=splits["train"])
    save_npz(calibration_path, x=splits["calibration"])
    save_npz(validation_path, x=splits["validation"])
    save_npz(test_path, x=splits["test"])
    save_npz(scaler_path, mean=mean, std=std)

    cfg_out = {
        "dataset": dataset,
        "raw_path": cfg_data.get("raw_path"),
        "window": {"length": spec.length, "stride": spec.stride},
        "splits": cfg_data["splits"],
        "scaler": {
            "eps": cfg_data["scaler"]["eps"],
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        },
        "shapes": {k: list(v.shape) for k, v in splits.items()},
    }
    save_yaml(config_path, cfg_out)

    return PreprocessOut(
        train_path=train_path,
        calibration_path=calibration_path,
        validation_path=validation_path,
        test_path=test_path,
        scaler_path=scaler_path,
        config_path=config_path,
    )
