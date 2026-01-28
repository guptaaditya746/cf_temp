from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .io import ensure_dir, load_yaml, save_yaml


@dataclass(frozen=True)
class RunPaths:
    root: Path
    artifacts: Path
    logs: Path


def resolve_run_dir(base_dir: str | Path, dataset: str, run_name: str) -> RunPaths:
    root = Path(base_dir) / dataset / run_name
    artifacts = root / "artifacts"
    logs = root / "logs"
    ensure_dir(artifacts)
    ensure_dir(logs)
    return RunPaths(root=root, artifacts=artifacts, logs=logs)


def load_config_file(path: str | Path) -> Dict[str, Any]:
    return load_yaml(path)


def freeze_config_to_run(cfg: Dict[str, Any], out_path: str | Path) -> None:
    save_yaml(out_path, cfg)


def get_default_config() -> Dict[str, Any]:
    # Minimal defaults; override via YAML or CLI.
    return {
        "base_dir": "runs",
        "seed": {"seed": 42, "deterministic_torch": True},
        "data": {
            "raw_path": None,  # dataset-specific resolver can fill this
            "feature_cols": None,  # if raw is a table; else ignored
            "window": {"length": 256, "stride": 16},
            "splits": {
                "train": 0.6,
                "calibration": 0.15,
                "validation": 0.1,
                "test": 0.15,
            },
            "scaler": {"eps": 1e-6},
        },
        "model": {
            "type": "conv1d_ae",
            "channels": [32, 64],
            "latent": 32,
            "kernel": 7,
            "dropout": 0.0,
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 30,
            "device": "cuda",  # falls back to cpu
        },
        "score": {
            "aggregation": "mean",  # for window_score: mean/max
        },
        "calibration": {
            "method": "quantile",
            "quantile": 0.995,
        },
        "select_targets": {
            "top_k": 50,
            "min_gap": 0,  # optional: enforce spacing in original time index if you track it
        },
        "counterfactuals": {
            "method": "nearest_normal_window",
            "normalcore": {"max_items": 2000, "subsample": "random"},
            "search": {"top_k_candidates": 200, "max_tries": 200},
        },
    }
