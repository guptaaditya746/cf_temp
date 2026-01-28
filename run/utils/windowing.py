from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    length: int
    stride: int


def make_windows(x: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """
    x: (T, F) -> windows: (N, L, F)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x shape (T,F), got {x.shape}")
    T, F = x.shape
    L, S = int(spec.length), int(spec.stride)
    if T < L:
        raise ValueError(f"T ({T}) < window length ({L})")
    starts = np.arange(0, T - L + 1, S, dtype=np.int64)
    N = starts.shape[0]
    w = np.empty((N, L, F), dtype=np.float32)
    for i, s in enumerate(starts):
        w[i] = x[s : s + L]
    return w


def split_windows(
    w: np.ndarray,
    splits: Dict[str, float],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    w: (N,L,F). Shuffles then splits by ratios.
    """
    if w.ndim != 3:
        raise ValueError(f"Expected (N,L,F), got {w.shape}")
    N = w.shape[0]
    keys = ["train", "calibration", "validation", "test"]
    ratios = np.array([splits[k] for k in keys], dtype=np.float64)
    if not np.isclose(ratios.sum(), 1.0):
        ratios = ratios / ratios.sum()

    rng = np.random.default_rng(int(seed))
    idx = np.arange(N, dtype=np.int64)
    rng.shuffle(idx)

    counts = (ratios * N).astype(int)
    # fix rounding to match N
    while counts.sum() < N:
        counts[np.argmax(ratios)] += 1
    while counts.sum() > N:
        counts[np.argmax(counts)] -= 1

    out = {}
    cur = 0
    for k, c in zip(keys, counts):
        sel = idx[cur : cur + c]
        out[k] = w[sel].astype(np.float32, copy=False)
        cur += c
    return out


def compute_scaler_params(
    train_w: np.ndarray, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean/std from train windows.
    train_w: (N,L,F)
    returns mean(F,), std(F,)
    """
    if train_w.ndim != 3:
        raise ValueError(f"Expected (N,L,F), got {train_w.shape}")
    # aggregate over N and L
    mean = train_w.mean(axis=(0, 1))
    var = train_w.var(axis=(0, 1))
    std = np.sqrt(var + float(eps))
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaling(w: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    w: (N,L,F) or (T,F)
    """
    mean = mean.reshape((1,) * (w.ndim - 1) + (-1,))
    std = std.reshape((1,) * (w.ndim - 1) + (-1,))
    return ((w - mean) / std).astype(np.float32, copy=False)
