from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..utils.io import load_npz, load_yaml, save_npz, save_yaml


@dataclass(frozen=True)
class SelectOut:
    targets_path: Path


def run_select_targets(run_dir: Path, cfg: Dict[str, Any]) -> SelectOut:
    art = run_dir / "artifacts"
    scores = load_npz(art / "test_scores.npz")["window"].astype(np.float32)
    test = load_npz(art / "test.npz")["x"].astype(np.float32)
    thr = load_yaml(art / "threshold.yaml")
    tau = float(thr["tau"])

    sel_cfg = cfg["select_targets"]
    top_k = int(sel_cfg["top_k"])

    # deterministic selection: sort by score descending, filter by score>tau
    idx = np.arange(scores.shape[0], dtype=np.int64)
    anomalous = idx[scores > tau]
    order = anomalous[np.argsort(scores[anomalous])[::-1]]

    chosen = order[:top_k]
    targets_x = test[chosen]

    targets_path = art / "targets.npz"
    save_npz(
        targets_path,
        idx=chosen.astype(np.int64),
        x=targets_x.astype(np.float32),
        score=scores[chosen].astype(np.float32),
    )

    save_yaml(
        art / "targets_selection.yaml",
        {
            "tau": tau,
            "rule": "score > tau, then top_k by descending score",
            "top_k": top_k,
            "n_anomalous": int(anomalous.shape[0]),
            "n_selected": int(chosen.shape[0]),
        },
    )

    print(
        f"[select_targets] selected {chosen.shape[0]}/{anomalous.shape[0]} windows (tau={tau:.6f})"
    )

    return SelectOut(targets_path=targets_path)
