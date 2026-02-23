from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.constraints import apply_constraints
from cftsad.core.scoring import reconstruction_score
from cftsad.methods.segment import detect_anomalous_segment
from cftsad.types import CFFailure, CFResult


def _z_normalize(segment: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(segment, axis=0, keepdims=True)
    std = np.std(segment, axis=0, keepdims=True)
    return (segment - mean) / (std + eps)


def _build_motif_index(normal_core: np.ndarray, motif_length: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    vectors = []
    sources: list[tuple[int, int]] = []
    K, L, _ = normal_core.shape

    if motif_length > L:
        return np.empty((0, motif_length), dtype=np.float64), sources

    for donor_idx in range(K):
        donor = normal_core[donor_idx]
        for start in range(0, L - motif_length + 1):
            seg = donor[start : start + motif_length]
            zn = _z_normalize(seg).reshape(-1)
            vectors.append(zn)
            sources.append((donor_idx, start))

    if not vectors:
        return np.empty((0, motif_length), dtype=np.float64), sources
    return np.asarray(vectors, dtype=np.float64), sources


def generate_motif(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray,
    threshold: float,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
    top_k: int = 5,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    score_before = reconstruction_score(model, x)

    segment = detect_anomalous_segment(model, x)
    if segment is None:
        return CFFailure(
            reason="segment_detection_failed",
            message="Unable to detect anomalous segment for motif substitution.",
            diagnostics={"score_before": score_before, "threshold": float(threshold)},
        )

    start, end = segment
    motif_length = end - start + 1

    motif_vectors, sources = _build_motif_index(normal_core, motif_length)
    if motif_vectors.shape[0] == 0:
        return CFFailure(
            reason="no_valid_cf",
            message="No motifs available for detected segment length.",
            diagnostics={"motif_length": motif_length, "score_before": score_before},
        )

    query = _z_normalize(x[start : end + 1]).reshape(-1).astype(np.float64)
    dists = np.linalg.norm(motif_vectors - query[None, :], axis=1)

    k = max(1, min(int(top_k), dists.shape[0]))
    top_idx = np.argpartition(dists, k - 1)[:k]
    top_idx = top_idx[np.argsort(dists[top_idx])]
    topk_distances = [float(dists[i]) for i in top_idx]

    best_result: CFResult | None = None
    best_score = np.inf
    best_source: tuple[int, int] | None = None

    for idx in top_idx:
        donor_idx, donor_start = sources[int(idx)]
        donor_end = donor_start + motif_length

        candidate = np.array(x, copy=True)
        candidate[start : end + 1] = normal_core[donor_idx, donor_start:donor_end]
        candidate = apply_constraints(candidate, x, immutable_features, bounds)

        score_after = reconstruction_score(model, candidate)
        if score_after <= threshold and score_after < best_score:
            best_score = score_after
            best_source = (donor_idx, donor_start)
            best_result = CFResult(
                x_cf=candidate,
                score_cf=score_after,
                meta={
                    "motif_length": motif_length,
                    "topk_distances": topk_distances,
                    "chosen_motif_source": {
                        "donor_idx": donor_idx,
                        "segment_start": donor_start,
                        "segment_end": donor_end - 1,
                    },
                    "score_before": score_before,
                    "score_after": score_after,
                },
            )

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    if best_result is not None:
        best_result.meta["runtime_ms"] = runtime_ms
        return best_result

    return CFFailure(
        reason="no_valid_cf",
        message="Top-k motifs did not produce a valid counterfactual.",
        diagnostics={
            "segment_start": start,
            "segment_end": end,
            "motif_length": motif_length,
            "topk_distances": topk_distances,
            "chosen_motif_source": best_source,
            "score_before": score_before,
            "threshold": float(threshold),
            "runtime_ms": runtime_ms,
        },
    )
