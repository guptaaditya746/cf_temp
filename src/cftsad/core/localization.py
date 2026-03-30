from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from cftsad.core.scoring import reconstruction_errors_per_timestep


def compute_timestep_error_reference(
    model: object,
    normal_core: np.ndarray | None,
) -> Dict[str, np.ndarray] | None:
    if normal_core is None:
        return None

    core = np.asarray(normal_core, dtype=np.float64)
    if core.ndim != 3 or core.shape[0] == 0:
        return None

    calib_errors = np.asarray(
        [reconstruction_errors_per_timestep(model, core[i]) for i in range(core.shape[0])],
        dtype=np.float64,
    )
    mean = np.mean(calib_errors, axis=0)
    std = np.std(calib_errors, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {
        "mean": mean,
        "std": std,
    }


def compute_localization_errors(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray] | None]:
    raw_errors = reconstruction_errors_per_timestep(model, x)
    if not normalize:
        return raw_errors, raw_errors, None

    reference = compute_timestep_error_reference(model, normal_core)
    if reference is None:
        return raw_errors, raw_errors, None

    normalized = (raw_errors - reference["mean"]) / reference["std"]
    return normalized, raw_errors, reference


def largest_contiguous_region(indices: np.ndarray) -> tuple[int, int] | None:
    if indices.size == 0:
        return None

    sorted_idx = np.sort(indices.astype(int))
    best_start = int(sorted_idx[0])
    best_end = int(sorted_idx[0])
    cur_start = int(sorted_idx[0])
    cur_end = int(sorted_idx[0])

    for idx in sorted_idx[1:]:
        val = int(idx)
        if val == cur_end + 1:
            cur_end = val
            continue
        if (cur_end - cur_start) > (best_end - best_start):
            best_start, best_end = cur_start, cur_end
        cur_start, cur_end = val, val

    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end

    return best_start, best_end


def centered_segment(
    center: int,
    length: int,
    series_length: int,
) -> tuple[int, int]:
    length = max(1, min(int(length), int(series_length)))
    half = length // 2
    start = max(0, int(center) - half)
    end = min(int(series_length) - 1, start + length - 1)
    start = max(0, end - length + 1)
    return int(start), int(end)


def detect_anomalous_segment(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray | None = None,
    *,
    top_fraction: float = 0.1,
    padding: int = 2,
    min_length: int | None = None,
    normalize_errors: bool = True,
) -> tuple[int, int] | None:
    L = x.shape[0]
    if L == 0:
        return None

    min_len = max(1, int(min_length) if min_length is not None else max(5, L // 20))
    loc_errors, _, _ = compute_localization_errors(
        model,
        x,
        normal_core=normal_core,
        normalize=normalize_errors,
    )

    top_frac = float(np.clip(top_fraction, 1.0 / max(1, L), 1.0))
    n_top = max(1, int(np.ceil(top_frac * L)))
    top_idx = np.argpartition(loc_errors, -n_top)[-n_top:]
    region = largest_contiguous_region(top_idx)
    if region is None:
        peak = int(np.argmax(loc_errors))
        return centered_segment(peak, min_len, L)

    start, end = region
    start = max(0, start - int(padding))
    end = min(L - 1, end + int(padding))
    cur_len = end - start + 1
    if cur_len < min_len:
        center = int(np.argmax(loc_errors[start : end + 1])) + start
        start, end = centered_segment(center, min_len, L)
    return int(start), int(end)


def detect_candidate_segments(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray | None = None,
    *,
    n_segments: int = 4,
    length_factors: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
    top_fraction: float = 0.1,
    padding: int = 2,
    min_length: int | None = None,
    normalize_errors: bool = True,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, Dict[str, np.ndarray] | None]:
    L = x.shape[0]
    if L == 0:
        return [], np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64), None

    loc_errors, raw_errors, reference = compute_localization_errors(
        model,
        x,
        normal_core=normal_core,
        normalize=normalize_errors,
    )
    min_len = max(1, int(min_length) if min_length is not None else max(5, L // 20))
    base = detect_anomalous_segment(
        model,
        x,
        normal_core=normal_core,
        top_fraction=top_fraction,
        padding=padding,
        min_length=min_len,
        normalize_errors=normalize_errors,
    )
    if base is None:
        return [], loc_errors, raw_errors, reference

    segments = {tuple(base)}
    base_len = int(base[1] - base[0] + 1)
    seed_count = max(int(n_segments), 4)
    seed_peaks = np.argsort(-loc_errors)[:seed_count]

    for peak in seed_peaks.tolist():
        for factor in length_factors:
            length = max(min_len, int(round(base_len * float(factor))))
            segments.add(centered_segment(int(peak), length, L))

    ranked = sorted(
        list(segments),
        key=lambda se: (
            -float(np.mean(loc_errors[se[0] : se[1] + 1])),
            -int(se[1] - se[0] + 1),
            int(se[0]),
        ),
    )
    return ranked[: max(1, int(n_segments))], loc_errors, raw_errors, reference


def build_segment_groups(
    segments: Sequence[tuple[int, int]],
    segment_scores: Dict[tuple[int, int], float],
    *,
    allow_pair_search: bool = False,
    max_pair_groups: int = 4,
) -> list[tuple[tuple[int, int], ...]]:
    singles = [(tuple(seg),) for seg in segments]
    if not allow_pair_search:
        return singles

    pair_groups: list[tuple[float, tuple[tuple[int, int], tuple[int, int]]]] = []
    ordered_segments = [tuple(seg) for seg in segments]
    for i, first in enumerate(ordered_segments):
        for second in ordered_segments[i + 1 :]:
            if not (first[1] < second[0] or second[1] < first[0]):
                continue
            ordered = tuple(sorted((first, second), key=lambda se: (se[0], se[1])))
            pair_score = float(segment_scores.get(ordered[0], 0.0) + segment_scores.get(ordered[1], 0.0))
            pair_groups.append((pair_score, ordered))

    pair_groups.sort(
        key=lambda item: (
            -item[0],
            -sum(seg[1] - seg[0] + 1 for seg in item[1]),
            item[1],
        )
    )
    return singles + [group for _, group in pair_groups[: max(0, int(max_pair_groups))]]
