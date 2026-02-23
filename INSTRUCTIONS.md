Below is the updated **`INSTRUCTIONS.md`** including the **Genetic (NSGA-II) multi-objective optimization approach**.
This defines the architecture, objectives, constraints, and API rules clearly for a public release.

---

# INSTRUCTIONS.md

## Counterfactual Explanations for Time-Series Anomaly Detection

### MSE-Based Scoring + Genetic Multi-Objective Optimization (v1.1)

---

# 1. Project Goal

This library generates **counterfactual explanations** for reconstruction-based time-series anomaly detection models.

Given:

* A trained reconstruction model
* An anomalous window ( x \in \mathbb{R}^{L \times F} )
* A normal core (set of normal windows)

The library returns:

> A minimally modified version of ( x ) whose reconstruction MSE falls below a threshold, while preserving realism and temporal coherence.

---

# 2. Unified Scoring (Mandatory Across All Methods)

All methods MUST use the same anomaly score:

### Reconstruction MSE Score

Per timestep:
[
e_t = \frac{1}{F} \sum_{f=1}^{F} (x_{t,f} - \hat{x}_{t,f})^2
]

Window score:
[
\text{score}(x) = \frac{1}{L} \sum_{t=1}^{L} e_t
]

This score defines:

* Anomaly detection
* Counterfactual validity
* Optimization target (for genetic)

---

# 3. Valid Counterfactual Definition

A counterfactual is valid if:

[
\text{score}(x_{cf}) \le \text{threshold}
]

Where threshold is:

* User-provided, OR
* Estimated from normal core (default 95% quantile)

---

# 4. Public API Contract

## 4.1 Entry Class

```python
CounterfactualExplainer(
    method: Literal["nearest", "segment", "motif", "genetic"],
    model: callable or torch module,
    normal_core: np.ndarray,  # shape (K, L, F)
    threshold: Optional[float] = None,
    **method_kwargs
)
```

---

## 4.2 Input Format

* Single window only
* Shape: `(L, F)`
* Type: `numpy.ndarray`
* No NaNs (v1.1)

---

## 4.3 Output Format

All methods MUST return:

### Success

```python
CFResult:
    x_cf: np.ndarray (L, F)
    score_cf: float
    meta: dict
```

### Failure

```python
CFFailure:
    reason: str
    message: str
    diagnostics: dict
```

Never return `None`.

---

# 5. Implemented Methods

---

# 5.1 Nearest Neighbour

* Replace full window with closest normal window.
* Distance metric: MSE.
* Deterministic.

---

# 5.2 Segment Substitution

* Detect anomalous segment.
* Replace with segment from nearest donor.
* Optional boundary smoothing.
* Deterministic.

---

# 5.3 Motif-Based Substitution

* Detect anomalous segment.
* Retrieve similar motifs from normal core.
* Replace only segment.
* Deterministic.

---

# 5.4 Genetic Multi-Objective Optimization (NSGA-II)

## Purpose

Instead of copying from existing normal data, this method **optimizes the input window directly in continuous space** to find a minimal modification that makes it normal.

---

# 6. Genetic Approach Design

---

## 6.1 Representation

Each individual in the population represents:

[
x_{cf} \in \mathbb{R}^{L \times F}
]

Optimization happens directly in input space.

---

## 6.2 Multi-Objective Optimization (NSGA-II)

We optimize:

### Objective 1 — Validity (Minimize anomaly score)

[
f_1 = \text{score}(x_{cf})
]

### Objective 2 — Proximity (Minimize distance to original)

[
f_2 = | x_{cf} - x |_2^2
]

### Objective 3 — Sparsity (Encourage minimal edits)

[
f_3 = \frac{\text{number of modified timesteps}}{L}
]

Optional (future):

* Smoothness penalty
* Spectral deviation
* PCA manifold distance

---

## 6.3 Hard Constraints (Mandatory)

Genetic individuals must satisfy:

* Immutable features unchanged
* Value bounds respected
* Optional physical constraints

Constraint violations increase feasibility penalty.

---

## 6.4 Soft Constraints (Optional Extension)

Can include:

* Temporal smoothness penalty
* PCA coupling preservation
* Spectral consistency

These influence Pareto ranking but do not hard-reject individuals.

---

## 6.5 Evolutionary Process

1. Initialize population around original window.
2. Evaluate objectives.
3. Apply NSGA-II:

   * Non-dominated sorting
   * Crowding distance
4. Apply crossover.
5. Apply mutation.
6. Repeat for `n_gen` generations.

---

## 6.6 Selection of Final Counterfactual

From Pareto front:

Select solution satisfying:

* score <= threshold
* minimal proximity

If none satisfy threshold:
Return best compromise with warning in meta.

---

## 6.7 Genetic Hyperparameters (Public)

```python
population_size = 100
n_generations = 50
crossover_rate = 0.9
mutation_rate = 0.1
seed = 42
```

Expose these via `method_kwargs`.

---

## 6.8 Meta Fields (Genetic)

* pareto_size
* best_objectives
* generations
* population_size
* constraint_violations
* runtime_ms

---

# 7. Threshold Handling

If threshold is None:

1. Compute score for each normal_core window.
2. Set threshold = 95% quantile.

---

# 8. Constraints (v1.1)

## 8.1 Immutable Features

User may specify:

```python
immutable_features = [2, 5]
```

These must remain unchanged.

---

## 8.2 Value Bounds

```python
bounds = {feature_index: (min_val, max_val)}
```

Applied to all methods.

---

## 8.3 Genetic Feasibility Handling

If constraint violated:

* Hard constraint → reject individual
* Soft constraint → penalty added

---

# 9. Package Structure

```
src/cftsad/
    __init__.py
    api.py
    types.py
    methods/
        nearest.py
        segment.py
        motif.py
        genetic.py
    core/
        scoring.py
        constraints.py
        distances.py
        evolution.py
```

---

# 10. Computational Complexity

| Method  | Complexity                |
| ------- | ------------------------- |
| Nearest | O(KLF)                    |
| Segment | O(KLF)                    |
| Motif   | O(K * motif_count)        |
| Genetic | O(pop * gen * model_eval) |

Genetic is significantly more expensive.

---

# 11. Error Handling

Possible failure reasons:

* invalid_input
* no_valid_cf
* constraint_violation
* optimization_failed
* segment_detection_failed

---

# 12. Reproducibility

All stochastic methods MUST expose:

```python
random_seed
```

Default = 42.

---

# 13. Version Scope

v1.1 includes:

* Nearest
* Segment
* Motif
* Genetic (NSGA-II)
* MSE scoring
* Hard constraints

Future versions may add:

* Generative infilling
* Diffusion models
* Advanced manifold constraints
* Batch inference
* GPU optimization support

---

# 14. Minimal Usage Example

```python
import numpy as np
from cftsad import CounterfactualExplainer

model = lambda x: x  # replace with real reconstructor
core = np.load("normal_core.npy")
x = np.load("window.npy")

explainer = CounterfactualExplainer(
    method="genetic",
    model=model,
    normal_core=core,
    threshold=None,
    population_size=100,
    n_generations=50,
)

result = explainer.explain(x)

print(result.score_cf)
```

---

# 15. Philosophy

This library unifies:

* Deterministic counterfactuals (nearest, segment, motif)
* Optimization-based counterfactuals (genetic)
* Consistent scoring
* Clear validity definition
* Strict API contract

The genetic method provides flexibility when no suitable donor exists, while deterministic methods ensure speed and interpretability.

---

END OF FILE
