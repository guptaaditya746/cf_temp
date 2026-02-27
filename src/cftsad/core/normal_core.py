from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cftsad.core.scoring import reconstruction_score


@dataclass
class NormalCoreBuildResult:
    normal_core: np.ndarray
    selected_indices: np.ndarray
    embeddings: np.ndarray
    reduced_embeddings: np.ndarray
    pca_components: np.ndarray
    pca_mean: np.ndarray
    selected_scores: np.ndarray
    all_scores: np.ndarray
    base_threshold: float
    strict_threshold: float


def _farthest_point_sampling(
    embeddings: np.ndarray,
    n_select: int,
    random_seed: int,
) -> np.ndarray:
    n = embeddings.shape[0]
    if n_select >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(random_seed))
    selected = np.empty(n_select, dtype=np.int64)
    selected[0] = int(rng.integers(0, n))

    min_d2 = np.sum((embeddings - embeddings[selected[0]]) ** 2, axis=1)
    for i in range(1, n_select):
        next_idx = int(np.argmax(min_d2))
        selected[i] = next_idx
        d2 = np.sum((embeddings - embeddings[next_idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    return np.sort(selected)


def build_normal_core(
    model: object,
    normal_core: np.ndarray,
    threshold: Optional[float],
    filter_factor: float = 1.0,
    threshold_quantile: float = 0.95,
    max_core_size: Optional[int] = None,
    use_diversity_sampling: bool = True,
    embedding_dim: Optional[int] = 32,
    random_seed: int = 42,
) -> NormalCoreBuildResult:
    core = np.asarray(normal_core, dtype=np.float64)
    n_windows = core.shape[0]

    all_scores = np.asarray(
        [reconstruction_score(model, core[i]) for i in range(n_windows)],
        dtype=np.float64,
    )

    if threshold is None:
        q = float(threshold_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("normal_core_threshold_quantile must be in (0, 1)")
        base_threshold = float(np.quantile(all_scores, q))
    else:
        base_threshold = float(threshold)

    strict_threshold = float(base_threshold) * float(filter_factor)
    keep_mask = all_scores <= strict_threshold
    keep_indices = np.where(keep_mask)[0]
    if keep_indices.size == 0:
        keep_indices = np.asarray([int(np.argmin(all_scores))], dtype=np.int64)

    filtered_core = core[keep_indices]
    filtered_scores = all_scores[keep_indices]
    filtered_embeddings = filtered_core.reshape(filtered_core.shape[0], -1)

    if max_core_size is not None:
        target_size = int(max_core_size)
        if target_size < 1:
            raise ValueError("normal_core_max_size must be >= 1 when provided")
        if filtered_core.shape[0] > target_size:
            if use_diversity_sampling:
                local_idx = _farthest_point_sampling(
                    filtered_embeddings,
                    n_select=target_size,
                    random_seed=int(random_seed),
                )
            else:
                local_idx = np.arange(target_size, dtype=np.int64)
            keep_indices = keep_indices[local_idx]
            filtered_core = filtered_core[local_idx]
            filtered_scores = filtered_scores[local_idx]
            filtered_embeddings = filtered_embeddings[local_idx]

    if embedding_dim is None:
        mean = np.mean(filtered_embeddings, axis=0, keepdims=True)
        reduced = np.asarray(filtered_embeddings - mean, dtype=np.float64)
        components = np.eye(filtered_embeddings.shape[1], dtype=np.float64)
    else:
        d = min(int(embedding_dim), int(filtered_embeddings.shape[1]))
        mean = np.mean(filtered_embeddings, axis=0, keepdims=True)
        centered = filtered_embeddings - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:d]
        reduced = centered @ components.T

    return NormalCoreBuildResult(
        normal_core=filtered_core,
        selected_indices=np.asarray(keep_indices, dtype=np.int64),
        embeddings=np.asarray(filtered_embeddings, dtype=np.float64),
        reduced_embeddings=np.asarray(reduced, dtype=np.float64),
        pca_components=np.asarray(components, dtype=np.float64),
        pca_mean=np.asarray(mean.reshape(-1), dtype=np.float64),
        selected_scores=np.asarray(filtered_scores, dtype=np.float64),
        all_scores=all_scores,
        base_threshold=float(base_threshold),
        strict_threshold=float(strict_threshold),
    )


def transform_embedding(
    windows: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    mean = np.asarray(pca_mean, dtype=np.float64).reshape(1, -1)
    comp = np.asarray(pca_components, dtype=np.float64)
    return (flat - mean) @ comp.T


def query_core_index(
    query_window: np.ndarray,
    reduced_embeddings: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    q = transform_embedding(
        query_window[np.newaxis, ...],
        pca_components=pca_components,
        pca_mean=pca_mean,
    )[0]
    d2 = np.sum((reduced_embeddings - q[np.newaxis, :]) ** 2, axis=1)
    k = max(1, min(int(top_k), int(d2.shape[0])))
    idx = np.argpartition(d2, k - 1)[:k]
    return idx[np.argsort(d2[idx])]
