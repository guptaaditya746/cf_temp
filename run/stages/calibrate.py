from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils.io import load_npz, save_yaml
from ..utils.model_io import load_torchscript
from ..utils.score import score_bundle


@dataclass(frozen=True)
class CalibrateOut:
    threshold_path: Path
    score_config_path: Path


def run_calibrate(run_dir: Path, cfg: Dict[str, Any]) -> CalibrateOut:
    art = run_dir / "artifacts"
    x = load_npz(art / "calibration.npz")["x"]

    mcfg = cfg["model"]
    device = (
        "cuda"
        if (str(mcfg.get("device", "cpu")) == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    model = load_torchscript(art / "model_ts.pt", device=device)

    scfg = cfg.get("score", {})
    agg = str(scfg.get("aggregation", "mean"))

    bs = int(mcfg.get("batch_size", 256))
    dl = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=bs, shuffle=False)

    scores = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            xh = model(xb)
            bundle = score_bundle(xb, xh, aggregation=agg)
            scores.append(bundle["window"].detach().cpu().numpy())
    scores = np.concatenate(scores, axis=0).astype(np.float32)

    ccfg = cfg["calibration"]
    method = str(ccfg["method"])

    if method == "quantile":
        q = float(ccfg["quantile"])
        tau = float(np.quantile(scores, q))
        meta = {"method": "quantile", "quantile": q}
    else:
        raise ValueError(
            f"Unsupported calibration.method: {method} (implement EVT here if needed)."
        )

    threshold_path = art / "threshold.yaml"
    save_yaml(
        threshold_path,
        {
            "tau": tau,
            "calibration": meta,
            "n_calibration": int(scores.shape[0]),
            "score_aggregation": agg,
        },
    )

    score_config_path = art / "score_config.yaml"
    save_yaml(score_config_path, {"aggregation": agg})

    print(f"[calibrate] tau={tau:.6f} method={method}")

    return CalibrateOut(
        threshold_path=threshold_path, score_config_path=score_config_path
    )
