from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils.io import load_npz, load_yaml, save_npz, save_yaml
from ..utils.model_io import load_torchscript
from ..utils.score import score_bundle


@dataclass(frozen=True)
class CFOut:
    normalcore_path: Path
    cf_results_path: Path
    cf_metadata_path: Path


def _build_normalcore(train_x: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    N = train_x.shape[0]
    if N <= max_items:
        return train_x.astype(np.float32, copy=False)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(N, size=int(max_items), replace=False)
    return train_x[idx].astype(np.float32, copy=False)


def _score_windows(
    model, device: str, x_np: np.ndarray, agg: str, batch_size: int
) -> np.ndarray:
    dl = DataLoader(
        TensorDataset(torch.from_numpy(x_np)), batch_size=batch_size, shuffle=False
    )
    scores = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            xh = model(xb)
            b = score_bundle(xb, xh, aggregation=agg)
            scores.append(b["window"].detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)


def run_counterfactuals(run_dir: Path, cfg: Dict[str, Any]) -> CFOut:
    art = run_dir / "artifacts"

    targets = load_npz(art / "targets.npz")
    x_t = targets["x"].astype(np.float32)  # (M,L,F)
    idx_t = targets["idx"].astype(np.int64)  # (M,)
    score_t = targets["score"].astype(np.float32)  # (M,)

    train_x = load_npz(art / "train.npz")["x"].astype(np.float32)
    thr = load_yaml(art / "threshold.yaml")
    tau = float(thr["tau"])

    mcfg = cfg["model"]
    device = (
        "cuda"
        if (str(mcfg.get("device", "cpu")) == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    model = load_torchscript(art / "model_ts.pt", device=device)

    agg = str(cfg.get("score", {}).get("aggregation", "mean"))
    bs = int(mcfg.get("batch_size", 256))

    # Build / save normalcore
    cf_cfg = cfg["counterfactuals"]
    max_items = int(cf_cfg["normalcore"]["max_items"])
    seed = int(cfg["seed"]["seed"])
    normalcore = _build_normalcore(train_x, max_items=max_items, seed=seed)

    normalcore_path = art / "normalcore.npz"
    save_npz(normalcore_path, x=normalcore)

    # Pre-score normalcore once
    normal_scores = _score_windows(model, device, normalcore, agg=agg, batch_size=bs)
    # Use the *most normal* candidates as the search pool (lowest scores)
    top_k_candidates = int(cf_cfg["search"]["top_k_candidates"])
    pool_idx = np.argsort(normal_scores)[: min(top_k_candidates, normalcore.shape[0])]
    pool = normalcore[pool_idx]  # (K,L,F)

    # Nearest-normal-window CF: choose candidate minimizing L2 distance to target
    # and validate it scores <= tau.
    M = x_t.shape[0]
    x_cf = np.empty_like(x_t, dtype=np.float32)
    cf_score = np.empty((M,), dtype=np.float32)
    chosen_pool = np.empty((M,), dtype=np.int64)
    success = np.zeros((M,), dtype=np.int64)

    # Compute distances in numpy (gradient-free)
    for i in range(M):
        xt = x_t[i]  # (L,F)
        # (K,L,F) -> (K,)
        d = ((pool - xt[None, :, :]) ** 2).mean(axis=(1, 2))
        order = np.argsort(d)

        # try candidates until one is below threshold
        found = False
        best_j = int(order[0])
        best_cf = pool[best_j]
        best_s = None

        for j in order[: int(cf_cfg["search"]["max_tries"])]:
            cand = pool[int(j)][None, :, :]  # (1,L,F)
            s = _score_windows(model, device, cand, agg=agg, batch_size=1)[0]
            if s <= tau:
                best_j = int(j)
                best_cf = pool[best_j]
                best_s = float(s)
                found = True
                break

        if best_s is None:
            # fallback: keep closest even if it fails threshold
            best_s = float(
                _score_windows(
                    model, device, best_cf[None, :, :], agg=agg, batch_size=1
                )[0]
            )

        x_cf[i] = best_cf
        cf_score[i] = np.float32(best_s)
        chosen_pool[i] = np.int64(best_j)
        success[i] = 1 if found else 0

    cf_results_path = art / "cf_results.npz"
    save_npz(
        cf_results_path,
        idx=idx_t,
        x_target=x_t,
        score_target=score_t,
        x_cf=x_cf,
        score_cf=cf_score,
        pool_choice=chosen_pool,
        success=success,
        tau=np.array([tau], dtype=np.float32),
    )

    cf_metadata_path = art / "cf_metadata.yaml"
    save_yaml(
        cf_metadata_path,
        {
            "method": str(cf_cfg["method"]),
            "tau": tau,
            "score_aggregation": agg,
            "normalcore": {"max_items": max_items, "used_pool_k": int(pool.shape[0])},
            "search": {
                "top_k_candidates": top_k_candidates,
                "max_tries": int(cf_cfg["search"]["max_tries"]),
                "success_rate": float(success.mean()) if success.size else 0.0,
            },
            "outputs": {
                "cf_results.npz": {
                    "x_target": list(x_t.shape),
                    "x_cf": list(x_cf.shape),
                }
            },
        },
    )

    print(
        f"[counterfactuals] success_rate={success.mean():.3f} (tau={tau:.6f}) wrote cf_results.npz + cf_metadata.yaml"
    )

    return CFOut(
        normalcore_path=normalcore_path,
        cf_results_path=cf_results_path,
        cf_metadata_path=cf_metadata_path,
    )
