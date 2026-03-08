from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from cftsad.core.constraints import apply_constraints
from cftsad.core.constraints_v2 import apply_constraints_v2
from cftsad.core.evolution import binary_tournament, nsga2_select
from cftsad.core.postprocess import build_explainability_meta
from cftsad.core.scoring import reconstruction_score
from cftsad.types import CFFailure, CFResult


def _score_candidate(
    model: object,
    x_cf: np.ndarray,
    score_fn: Optional[Callable[[np.ndarray], float]],
) -> float:
    if score_fn is not None:
        return float(score_fn(np.asarray(x_cf, dtype=np.float64)))
    return reconstruction_score(model, x_cf)


def _reconstruct_with_model(model: object, x: np.ndarray) -> np.ndarray:
    try:
        y = model(x)
    except Exception:
        y = model(x[np.newaxis, ...])

    y = np.asarray(y)
    if y.shape == x.shape:
        return y
    if y.ndim == 3 and y.shape[0] == 1 and y.shape[1:] == x.shape:
        return y[0]
    raise ValueError(
        f"Model reconstruction shape mismatch in genetic method: {y.shape} vs {x.shape}"
    )


def _constraint_violation(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    immutable_features: tuple[int, ...],
    bounds: Dict[int, Tuple[float, float]],
    use_constraints_v2: bool,
    max_delta_per_step: Optional[float],
    relational_linear: Optional[Dict[str, tuple[int, int, float]]],
) -> tuple[float, dict]:
    if use_constraints_v2:
        _, v, breakdown = apply_constraints_v2(
            x_cf=x_cf,
            x_original=x_original,
            immutable_features=immutable_features,
            bounds=bounds,
            max_delta_per_step=max_delta_per_step,
            relational_linear=relational_linear,
        )
        return float(v), breakdown

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

    return float(violation), {"legacy": float(violation)}


def _sparsity_objective(
    x_cf: np.ndarray, x_original: np.ndarray, eps: float = 1e-8
) -> float:
    changed = np.any(np.abs(x_cf - x_original) > eps, axis=1)
    return float(np.mean(changed.astype(np.float64)))


def _smoothness_objective(x_cf: np.ndarray) -> float:
    if x_cf.shape[0] <= 1:
        return 0.0
    d1 = np.diff(x_cf, axis=0)
    return float(np.sum(d1 * d1))


def _plausibility_objective(x_cf: np.ndarray, normal_core: np.ndarray) -> float:
    d2 = np.mean((normal_core - x_cf[np.newaxis, :, :]) ** 2, axis=(1, 2))
    return float(np.min(d2))


def _evaluate_population(
    model: object,
    population: np.ndarray,
    x_original: np.ndarray,
    normal_core: np.ndarray,
    immutable_features: tuple[int, ...],
    bounds: Dict[int, Tuple[float, float]],
    include_smoothness_objective: bool,
    use_plausibility_objective: bool,
    threshold: float,
    validity_margin: float,
    use_constraints_v2: bool,
    max_delta_per_step: Optional[float],
    relational_linear: Optional[Dict[str, tuple[int, int, float]]],
    score_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    n = population.shape[0]
    n_obj = 3 + int(include_smoothness_objective) + int(use_plausibility_objective)
    objectives = np.zeros((n, n_obj), dtype=np.float64)
    violations = np.zeros(n, dtype=np.float64)
    breakdowns: list[dict] = []
    thr_target = float(threshold) - float(validity_margin)

    for i in range(n):
        x_cf = population[i]
        c_v, c_break = _constraint_violation(
            x_cf,
            x_original,
            immutable_features,
            bounds,
            use_constraints_v2=use_constraints_v2,
            max_delta_per_step=max_delta_per_step,
            relational_linear=relational_linear,
        )
        score = _score_candidate(model, x_cf, score_fn)
        validity_violation = max(0.0, float(score) - thr_target)
        violations[i] = float(c_v + validity_violation)
        c_break = dict(c_break)
        c_break["validity"] = float(validity_violation)
        breakdowns.append(c_break)

        diff = x_cf - x_original
        objectives[i, 0] = float(score)
        objectives[i, 1] = float(np.sum(diff * diff))
        objectives[i, 2] = _sparsity_objective(x_cf, x_original)
        col = 3
        if include_smoothness_objective:
            objectives[i, col] = _smoothness_objective(x_cf)
            col += 1
        if use_plausibility_objective:
            objectives[i, col] = _plausibility_objective(x_cf, normal_core)

    return objectives, violations, breakdowns


def _initialize_population(
    x: np.ndarray,
    normal_core: np.ndarray,
    population_size: int,
    rng: np.random.Generator,
    immutable_features: tuple[int, ...],
    bounds: Dict[int, Tuple[float, float]],
    model: object,
    mutation_sigma: float,
) -> np.ndarray:
    l_steps, n_features = x.shape
    pop = np.repeat(x[np.newaxis, :, :], population_size, axis=0)

    core_std = np.std(normal_core.reshape(-1, n_features), axis=0)
    core_std = np.where(core_std < 1e-8, 1e-3, core_std)

    pop[0] = np.array(x, copy=True)
    if population_size >= 2:
        dists = np.mean((normal_core - x[np.newaxis, :, :]) ** 2, axis=(1, 2))
        donor_idx = int(np.argmin(dists))
        donor = np.array(normal_core[donor_idx], copy=True)
        pop[1] = apply_constraints(donor, x, immutable_features, bounds)
    if population_size >= 3:
        recon = _reconstruct_with_model(model, x)
        pop[2] = apply_constraints(recon, x, immutable_features, bounds)

    start_i = 3 if population_size >= 3 else population_size
    sigma = max(float(mutation_sigma), 1e-8)
    for i in range(start_i, population_size):
        noise = rng.normal(0.0, 1.0, size=(l_steps, n_features)) * (
            sigma * core_std[np.newaxis, :]
        )
        candidate = x + noise
        pop[i] = apply_constraints(candidate, x, immutable_features, bounds)
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


def _gaussian_mutation(
    x_cf: np.ndarray,
    mutation_rate: float,
    normal_core: np.ndarray,
    mutation_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.array(x_cf, copy=True)
    l_steps, n_features = out.shape
    core_std = np.std(normal_core.reshape(-1, n_features), axis=0)
    core_std = np.where(core_std < 1e-8, 1e-3, core_std)
    mask = rng.random(size=(l_steps, n_features)) < mutation_rate
    noise = rng.normal(0.0, 1.0, size=(l_steps, n_features)) * (
        float(mutation_sigma) * core_std[np.newaxis, :]
    )
    out[mask] += noise[mask]
    return out


def _segment_copy_mutation(
    x_cf: np.ndarray,
    normal_core: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.array(x_cf, copy=True)
    l_steps = out.shape[0]
    donor_idx = int(rng.integers(0, normal_core.shape[0]))
    seg_len = int(rng.integers(max(2, l_steps // 20), max(3, l_steps // 5) + 1))
    start = int(rng.integers(0, max(1, l_steps - seg_len + 1)))
    end = min(l_steps, start + seg_len)
    out[start:end] = normal_core[donor_idx, start:end]
    return out


def _motif_insert_mutation(
    x_cf: np.ndarray,
    normal_core: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.array(x_cf, copy=True)
    l_steps = out.shape[0]
    donor_idx = int(rng.integers(0, normal_core.shape[0]))
    motif_len = int(rng.integers(max(3, l_steps // 30), max(4, l_steps // 8) + 1))
    if motif_len >= l_steps:
        motif_len = max(2, l_steps - 1)
    donor_start = int(rng.integers(0, max(1, l_steps - motif_len + 1)))
    target_start = int(rng.integers(0, max(1, l_steps - motif_len + 1)))
    motif = normal_core[donor_idx, donor_start : donor_start + motif_len]
    out[target_start : target_start + motif_len] = motif
    return out


def _mutate(
    x_cf: np.ndarray,
    mutation_rate: float,
    normal_core: np.ndarray,
    mutation_sigma: float,
    rng: np.random.Generator,
    structured_mutation_weight: float,
) -> np.ndarray:
    if rng.random() < float(structured_mutation_weight):
        if rng.random() < 0.5:
            return _segment_copy_mutation(x_cf, normal_core, rng)
        return _motif_insert_mutation(x_cf, normal_core, rng)
    return _gaussian_mutation(x_cf, mutation_rate, normal_core, mutation_sigma, rng)


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
    mutation_sigma: float = 0.05,
    use_smoothness_objective: bool = False,
    use_plausibility_objective: bool = True,
    structured_mutation_weight: float = 0.35,
    validity_margin: float = 0.0,
    top_m_solutions: int = 5,
    early_stop_patience: int = 15,
    use_constraints_v2: bool = True,
    max_delta_per_step: Optional[float] = None,
    relational_linear: Optional[Dict[str, tuple[int, int, float]]] = None,
    score_fn: Optional[Callable[[np.ndarray], float]] = None,
    random_seed: int = 42,
) -> CFResult | CFFailure:
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(random_seed))
    thr = float(threshold)

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
        model=model,
        mutation_sigma=float(mutation_sigma),
    )

    objectives, violations, breakdowns = _evaluate_population(
        model=model,
        population=population,
        x_original=x,
        normal_core=normal_core,
        immutable_features=immutable_tuple,
        bounds=bounds_dict,
        include_smoothness_objective=bool(use_smoothness_objective),
        use_plausibility_objective=bool(use_plausibility_objective),
        threshold=thr,
        validity_margin=float(validity_margin),
        use_constraints_v2=bool(use_constraints_v2),
        max_delta_per_step=max_delta_per_step,
        relational_linear=relational_linear,
        score_fn=score_fn,
    )

    history_best_validity = []
    history_best_proximity = []
    best_valid_score = np.inf
    patience_left = int(early_stop_patience)

    for _gen in range(int(n_generations)):
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
            c1 = _mutate(
                c1,
                float(mutation_rate),
                normal_core,
                float(mutation_sigma),
                rng,
                structured_mutation_weight=float(structured_mutation_weight),
            )
            c2 = _mutate(
                c2,
                float(mutation_rate),
                normal_core,
                float(mutation_sigma),
                rng,
                structured_mutation_weight=float(structured_mutation_weight),
            )
            c1 = apply_constraints(c1, x, immutable_tuple, bounds_dict)
            c2 = apply_constraints(c2, x, immutable_tuple, bounds_dict)
            children.extend([c1, c2])

        offspring = np.asarray(children[: population.shape[0]], dtype=np.float64)
        combined = np.concatenate([population, offspring], axis=0)
        combined_obj, combined_viol, combined_breakdowns = _evaluate_population(
            model=model,
            population=combined,
            x_original=x,
            normal_core=normal_core,
            immutable_features=immutable_tuple,
            bounds=bounds_dict,
            include_smoothness_objective=bool(use_smoothness_objective),
            use_plausibility_objective=bool(use_plausibility_objective),
            threshold=thr,
            validity_margin=float(validity_margin),
            use_constraints_v2=bool(use_constraints_v2),
            max_delta_per_step=max_delta_per_step,
            relational_linear=relational_linear,
            score_fn=score_fn,
        )

        keep_idx, _, _, _ = nsga2_select(
            combined_obj,
            combined_viol,
            target_size=population.shape[0],
        )
        population = combined[keep_idx]
        objectives = combined_obj[keep_idx]
        violations = combined_viol[keep_idx]
        breakdowns = [combined_breakdowns[int(i)] for i in keep_idx]

        best_i = int(np.argmin(objectives[:, 0]))
        best_score = float(objectives[best_i, 0])
        history_best_validity.append(best_score)
        history_best_proximity.append(float(objectives[best_i, 1]))
        if best_score + 1e-12 < best_valid_score:
            best_valid_score = best_score
            patience_left = int(early_stop_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    _, _, _, fronts = nsga2_select(objectives, violations, target_size=population.shape[0])
    pareto_front = fronts[0] if fronts else []
    if not pareto_front:
        return CFFailure(
            reason="optimization_failed",
            message="No Pareto front could be produced.",
            diagnostics={
                "population_size": int(population_size),
                "n_generations": int(n_generations),
            },
        )

    pareto_idx = np.asarray(pareto_front, dtype=np.int64)
    pareto_pop = population[pareto_idx]
    pareto_obj = objectives[pareto_idx]
    pareto_viol = violations[pareto_idx]
    pareto_break = [breakdowns[int(i)] for i in pareto_idx]

    feasible_mask = pareto_viol <= 1e-12
    valid_mask = feasible_mask & (pareto_obj[:, 0] <= thr)
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        local_order = np.argsort(pareto_obj[valid_indices, 1])
        ordered_valid = valid_indices[local_order]
    else:
        ordered_valid = np.array([], dtype=np.int64)

    if ordered_valid.size > 0:
        best_local = int(ordered_valid[0])
        warning = None
        selection_mode = "valid"
    else:
        warning = "No feasible threshold-valid solution; returning best compromise."
        compromise = np.lexsort((pareto_obj[:, 1], pareto_obj[:, 0], pareto_viol))
        best_local = int(compromise[0])
        selection_mode = "best_compromise"

    best_cf = pareto_pop[best_local]
    best_obj = pareto_obj[best_local]
    best_viol = float(pareto_viol[best_local])
    best_breakdown = pareto_break[best_local]

    m = max(1, int(top_m_solutions))
    top_solutions = []
    source_idx = (
        ordered_valid
        if ordered_valid.size > 0
        else np.lexsort((pareto_obj[:, 1], pareto_obj[:, 0], pareto_viol))
    )
    for i in source_idx[:m]:
        top_solutions.append(
            {
                "score_cf": float(pareto_obj[int(i), 0]),
                "proximity": float(pareto_obj[int(i), 1]),
                "sparsity": float(pareto_obj[int(i), 2]),
                "violation": float(pareto_viol[int(i)]),
            }
        )

    objective_names = ["validity", "proximity", "sparsity"]
    col = 3
    if bool(use_smoothness_objective):
        objective_names.append("smoothness")
        col += 1
    if bool(use_plausibility_objective):
        objective_names.append("plausibility")

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    meta = {
        "pareto_size": int(len(pareto_front)),
        "best_objectives": [float(v) for v in best_obj.tolist()],
        "best_objective_names": objective_names,
        "top_solutions": top_solutions,
        "generations": int(n_generations),
        "population_size": int(population_size),
        "constraint_violation": best_viol,
        "constraint_breakdown": best_breakdown,
        "runtime_ms": runtime_ms,
        "warning": warning,
        "threshold": thr,
        "valid": bool(best_obj[0] <= thr and best_viol <= 1e-12),
        "threshold_valid": bool(best_obj[0] <= thr),
        "constraints_feasible": bool(best_viol <= 1e-12),
        "valid_counterfactual_found": bool(ordered_valid.size > 0),
        "returned_best_compromise": bool(ordered_valid.size == 0),
        "selection_mode": selection_mode,
        "score_source": "custom" if score_fn is not None else "reconstruction_score",
        "mutation_sigma": float(mutation_sigma),
        "structured_mutation_weight": float(structured_mutation_weight),
        "use_plausibility_objective": bool(use_plausibility_objective),
        "history_best_validity": [float(v) for v in history_best_validity],
        "history_best_proximity": [float(v) for v in history_best_proximity],
        "random_seed": int(random_seed),
    }
    meta.update(build_explainability_meta(x, best_cf))
    return CFResult(
        x_cf=best_cf,
        score_cf=float(best_obj[0]),
        meta=meta,
    )
