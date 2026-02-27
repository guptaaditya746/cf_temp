from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_core_artifacts(
    path: str | Path,
    *,
    normal_core: np.ndarray,
    selected_indices: np.ndarray,
    embeddings: np.ndarray,
    selected_scores: np.ndarray,
    all_scores: np.ndarray,
    reduced_embeddings: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    pca_mean: np.ndarray | None = None,
    metadata: Dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "normal_core": np.asarray(normal_core, dtype=np.float64),
        "selected_indices": np.asarray(selected_indices, dtype=np.int64),
        "embeddings": np.asarray(embeddings, dtype=np.float64),
        "selected_scores": np.asarray(selected_scores, dtype=np.float64),
        "all_scores": np.asarray(all_scores, dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if reduced_embeddings is not None:
        payload["reduced_embeddings"] = np.asarray(reduced_embeddings, dtype=np.float64)
    if pca_components is not None:
        payload["pca_components"] = np.asarray(pca_components, dtype=np.float64)
    if pca_mean is not None:
        payload["pca_mean"] = np.asarray(pca_mean, dtype=np.float64)
    np.savez_compressed(target, **payload)


def load_core_artifacts(path: str | Path) -> Dict[str, Any]:
    blob = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(blob["metadata_json"]))
    return {
        "normal_core": blob["normal_core"],
        "selected_indices": blob["selected_indices"],
        "embeddings": blob["embeddings"],
        "reduced_embeddings": blob["reduced_embeddings"]
        if "reduced_embeddings" in blob
        else None,
        "pca_components": blob["pca_components"] if "pca_components" in blob else None,
        "pca_mean": blob["pca_mean"] if "pca_mean" in blob else None,
        "selected_scores": blob["selected_scores"],
        "all_scores": blob["all_scores"],
        "metadata": metadata,
    }
