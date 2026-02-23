from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.constraints import apply_constraints
from cftsad.core.evolution import binary_tournament, nsga2_select
from cftsad.core.scoring import reconstruction_score
from cftsad.types import CFFailure, CFResult


def _constraint_violation(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    immutable_features: tuple[int, ...],
    bounds: Dict[int, Tuple[float, float]],
) -> float:
    violation = 0.0

    if immutable_features:
        for feat in immutable_features:
            idx = int(feat)
            violation += float(np.sum(np.abs(x_cf[:, idx] - x_original[:, idx])))

    if bounds:
        for feat, (lo, hi) in bounds.items():
            idx = int(feat)
            lo_v = float(lo)
            hi_v = float(hi)
            below = np.maximum(0.0, lo_v - x_cf[:, idx])
            above = np.maximum(0.0, x_cf[:, idx] - hi_v)
            violation += float(np.sum(below + above))

    return float(violation)


def _sparsity_objective(x_cf: np.ndarray, x_original: np.ndarray, eps: float = 1e-8) -> float:
    changed = np.any(np.abs(x_cf - x_original) > eps, axis=1)
    return float(np.mean(changed.astype(np.float64)))


def _evaluate_population(
    model: object,
    population: np.ndarray,
    x_original: np.ndarray,
    immutable_features: tuple[int, ...],
    bounds: Dict[int, Tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    n = population.shape[0]
    objectives = np.zeros((n, 3), dtype=np.float64)
    violations = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x_cf = population[i]
        violations[i] = _constraint_violation(
            x_cf,
            x_original,
            immutable_features,
            bounds,
        )
        objectives[i, 0] = reconstruction_score(model, x_cf)
        diff = x_cf - x_original
        objectives[i, 1] = float(np.sum(diff * diff))
        objectives[i, 2] = _sparsity_objective(x_cf, x_original)

    return objectives, violations


def _initialize_population(
    x: np.ndarray,
    normal_core: np.ndarray,
    population_size: int,
    rng: np.random.Generator,
    immutable_features: tuple[int, ...],
    bounds: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    L, F = x.shape
    pop = np.repeat(x[np.newaxis, :, :], population_size, axis=0)

    core_std = np.std(normal_core.reshape(-1, F), axis=0)
    core_std = np.where(core_std < 1e-8, 1e-3, core_std)

    for i in range(population_size):
        noise_scale = rng.uniform(0.01, 0.2)
        noise = rng.normal(0.0, 1.0, size=(L, F)) * core_std[np.newaxis, :] * noise_scale
        candidate = x + noise
        candidate = apply_constraints(candidate, x, immutable_features, bounds)
        pop[i] = candidate

    pop[0] = np.array(x, copy=True)
    return pop


def _crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() >= crossover_rate:
        return np.array(p1, copy=True), np.array(p2, copy=True)

    alpha = rng.uniform(0.0, 1.0)
    c1 = alpha * p1 + (1.0 - alpha) * p2
    c2 = alpha * p2 + (1.0 - alpha) * p1
    return c1, c2


def _mutate(
    x_cf: np.ndarray,
    mutation_rate: float,
    normal_core: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.array(x_cf, copy=True)
    L, F = out.shape
    core_std = np.std(normal_core.reshape(-1, F), axis=0)
    core_std = np.where(core_std < 1e-8, 1e-3, core_std)

    mask = rng.random(size=(L, F)) < mutation_rate
    noise = rng.normal(0.0, 1.0, size=(L, F)) * (0.05 * core_std[np.newaxis, :])
    out[mask] += noise[mask]
    return out


def generate_genetic(
    model: object,
    x: np.ndarray,
    normal_core: np.ndarray,
    threshold: float,
    immutable_features: Optional[Iterable[int]] = None,
    bounds: Optional[Dict[int, Tuple[float, float]]] = None,
    population_size: int = 100,
    n_generations: int = 50,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    random_seed: int = 42,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(random_seed))

    immutable_tuple = tuple(int(v) for v in (immutable_features or ()))
    bounds_dict = dict(bounds or {})

    if population_size < 4:
        return CFFailure(
            reason="invalid_input",
            message="population_size must be >= 4",
            diagnostics={"population_size": int(population_size)},
        )
    if n_generations < 1:
        return CFFailure(
            reason="invalid_input",
            message="n_generations must be >= 1",
            diagnostics={"n_generations": int(n_generations)},
        )

    population = _initialize_population(
        x=x,
        normal_core=normal_core,
        population_size=int(population_size),
        rng=rng,
        immutable_features=immutable_tuple,
        bounds=bounds_dict,
    )

    objectives, violations = _evaluate_population(
        model,
        population,
        x,
        immutable_tuple,
        bounds_dict,
    )

    for _ in range(int(n_generations)):
        selected_idx, ranks, crowding, _ = nsga2_select(
            objectives,
            violations,
            target_size=population.shape[0],
        )
        mating_pool = population[selected_idx]
        mating_ranks = ranks[selected_idx]
        mating_crowding = crowding[selected_idx]

        children = []
        while len(children) < population.shape[0]:
            i1 = binary_tournament(mating_ranks, mating_crowding, rng)
            i2 = binary_tournament(mating_ranks, mating_crowding, rng)

            p1 = mating_pool[i1]
            p2 = mating_pool[i2]

            c1, c2 = _crossover(p1, p2, float(crossover_rate), rng)
            c1 = _mutate(c1, float(mutation_rate), normal_core, rng)
            c2 = _mutate(c2, float(mutation_rate), normal_core, rng)

            c1 = apply_constraints(c1, x, immutable_tuple, bounds_dict)
            c2 = apply_constraints(c2, x, immutable_tuple, bounds_dict)
            children.extend([c1, c2])

        offspring = np.asarray(children[: population.shape[0]], dtype=np.float64)

        combined = np.concatenate([population, offspring], axis=0)
        combined_obj, combined_viol = _evaluate_population(
            model,
            combined,
            x,
            immutable_tuple,
            bounds_dict,
        )

        keep_idx, _, _, _ = nsga2_select(
            combined_obj,
            combined_viol,
            target_size=population.shape[0],
        )
        population = combined[keep_idx]
        objectives = combined_obj[keep_idx]
        violations = combined_viol[keep_idx]

    _, _, _, fronts = nsga2_select(objectives, violations, target_size=population.shape[0])
    pareto_front = fronts[0] if fronts else []

    if not pareto_front:
        return CFFailure(
            reason="optimization_failed",
            message="No Pareto front could be produced.",
            diagnostics={"population_size": int(population_size), "n_generations": int(n_generations)},
        )

    pareto_idx = np.asarray(pareto_front, dtype=np.int64)
    pareto_pop = population[pareto_idx]
    pareto_obj = objectives[pareto_idx]
    pareto_viol = violations[pareto_idx]

    feasible_mask = pareto_viol <= 1e-12
    valid_mask = feasible_mask & (pareto_obj[:, 0] <= float(threshold))

    warning = None
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        best_local = valid_indices[np.argmin(pareto_obj[valid_indices, 1])]
    else:
        # Best compromise if no threshold-valid solution exists.
        warning = "No Pareto solution satisfied threshold; returning best compromise."
        compromise = np.lexsort((pareto_obj[:, 1], pareto_obj[:, 0]))
        best_local = int(compromise[0])

    best_cf = pareto_pop[best_local]
    best_obj = pareto_obj[best_local]
    best_viol = pareto_viol[best_local]

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return CFResult(
        x_cf=best_cf,
        score_cf=float(best_obj[0]),
        meta={
            "pareto_size": int(len(pareto_front)),
            "best_objectives": {
                "validity": float(best_obj[0]),
                "proximity": float(best_obj[1]),
                "sparsity": float(best_obj[2]),
            },
            "generations": int(n_generations),
            "population_size": int(population_size),
            "constraint_violations": float(best_viol),
            "runtime_ms": runtime_ms,
            "warning": warning,
            "threshold": float(threshold),
            "valid": bool(best_obj[0] <= float(threshold)),
        },
    )
