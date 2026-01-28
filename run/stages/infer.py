from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils.io import load_npz, save_npz
from ..utils.model_io import load_torchscript
from ..utils.score import score_bundle


@dataclass(frozen=True)
class InferOut:
    test_scores_path: Path
    error_maps_path: Path
    reconstructions_path: Path


def run_infer(run_dir: Path, cfg: Dict[str, Any]) -> InferOut:
    art = run_dir / "artifacts"
    x = load_npz(art / "test.npz")["x"]

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

    all_window = []
    all_time = []
    all_feature = []
    all_E = []
    all_xhat = []

    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            xh = model(xb)
            bundle = score_bundle(xb, xh, aggregation=agg)

            all_window.append(bundle["window"].detach().cpu().numpy())
            all_time.append(bundle["time"].detach().cpu().numpy())
            all_feature.append(bundle["feature"].detach().cpu().numpy())
            all_E.append(bundle["E"].detach().cpu().numpy())
            all_xhat.append(xh.detach().cpu().numpy())

    test_scores_path = art / "test_scores.npz"
    error_maps_path = art / "error_maps.npz"
    reconstructions_path = art / "reconstructions.npz"

    save_npz(
        test_scores_path,
        window=np.concatenate(all_window, axis=0).astype(np.float32),
        time=np.concatenate(all_time, axis=0).astype(np.float32),
        feature=np.concatenate(all_feature, axis=0).astype(np.float32),
    )
    save_npz(error_maps_path, E=np.concatenate(all_E, axis=0).astype(np.float32))
    save_npz(
        reconstructions_path, x_hat=np.concatenate(all_xhat, axis=0).astype(np.float32)
    )

    print(f"[infer] wrote scores/E/recons for N={x.shape[0]} windows")

    return InferOut(
        test_scores_path=test_scores_path,
        error_maps_path=error_maps_path,
        reconstructions_path=reconstructions_path,
    )
