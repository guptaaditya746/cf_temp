from __future__ import annotations

import numpy as np


def _constrained_dominates(
    i: int,
    j: int,
    objectives: np.ndarray,
    violations: np.ndarray,
) -> bool:
    vi = violations[i]
    vj = violations[j]

    if vi == 0.0 and vj > 0.0:
        return True
    if vi > 0.0 and vj == 0.0:
        return False
    if vi > 0.0 and vj > 0.0:
        return vi < vj

    oi = objectives[i]
    oj = objectives[j]
    return np.all(oi <= oj) and np.any(oi < oj)


def fast_non_dominated_sort(
    objectives: np.ndarray,
    violations: np.ndarray,
) -> tuple[list[list[int]], np.ndarray]:
    n = objectives.shape[0]
    dominates = [set() for _ in range(n)]
    dominated_count = np.zeros(n, dtype=np.int64)
    ranks = np.full(n, -1, dtype=np.int64)

    fronts: list[list[int]] = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _constrained_dominates(i, j, objectives, violations):
                dominates[i].add(j)
            elif _constrained_dominates(j, i, objectives, violations):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    f = 0
    while f < len(fronts) and fronts[f]:
        next_front: list[int] = []
        for i in fronts[f]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    ranks[j] = f + 1
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        f += 1

    return fronts, ranks


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    distances = np.zeros(len(front), dtype=np.float64)
    if len(front) <= 2:
        distances[:] = np.inf
        return distances

    front_obj = objectives[np.asarray(front)]
    n_obj = front_obj.shape[1]

    for m in range(n_obj):
        order = np.argsort(front_obj[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        min_v = front_obj[order[0], m]
        max_v = front_obj[order[-1], m]
        denom = max(max_v - min_v, 1e-12)

        for pos in range(1, len(front) - 1):
            left = front_obj[order[pos - 1], m]
            right = front_obj[order[pos + 1], m]
            distances[order[pos]] += (right - left) / denom

    return distances


def nsga2_select(
    objectives: np.ndarray,
    violations: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    fronts, ranks = fast_non_dominated_sort(objectives, violations)

    selected: list[int] = []
    crowding = np.zeros(objectives.shape[0], dtype=np.float64)

    for front in fronts:
        if not front:
            continue
        front_cd = crowding_distance(objectives, front)
        for local_idx, global_idx in enumerate(front):
            crowding[global_idx] = front_cd[local_idx]

        if len(selected) + len(front) <= target_size:
            selected.extend(front)
            continue

        need = target_size - len(selected)
        order = np.argsort(-front_cd)
        selected.extend([front[i] for i in order[:need]])
        break

    return np.asarray(selected, dtype=np.int64), ranks, crowding, fronts


def binary_tournament(
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
) -> int:
    i, j = rng.integers(0, len(ranks), size=2)
    if ranks[i] < ranks[j]:
        return int(i)
    if ranks[j] < ranks[i]:
        return int(j)
    if crowding[i] > crowding[j]:
        return int(i)
    if crowding[j] > crowding[i]:
        return int(j)
    return int(i if rng.random() < 0.5 else j)
