Below is your **updated and production-ready `INSTRUCTIONS.md`** including:

* **COMTE (Segment Substitution)**
* **Generative (Mask + Infilling)**
* Unified MSE scoring
* Constraints
* Clear API contract
* Deterministic + stochastic behavior definitions

This version assumes you are shipping:

> nearest, comte, generative, motif, genetic (optional)

---

# INSTRUCTIONS.md

## Counterfactual Explanations for Time-Series Anomaly Detection

### Reconstruction-Based Models (MSE Scoring) — v1.2

---

# 1. Project Objective

This library provides **counterfactual explanations** for reconstruction-based time-series anomaly detection models.

Given:

* A trained reconstruction model
* An anomalous window ( x \in \mathbb{R}^{L \times F} )
* A set of normal windows (normal core)

The system generates:

> A minimally modified version of ( x ) such that its reconstruction MSE falls below a defined anomaly threshold, while preserving temporal coherence and realism.

---

# 2. Unified Anomaly Scoring (MANDATORY)

All methods MUST use the same anomaly score.

## 2.1 Reconstruction MSE

For input ( x ) and reconstruction ( \hat{x} ):

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
* Optimization objective (genetic)
* Candidate ranking (comte + generative)

---

# 3. Valid Counterfactual Definition

A counterfactual is valid if:

[
\text{score}(x_{cf}) \le \text{threshold}
]

Threshold is:

* User-provided OR
* Computed from normal core (default: 95th percentile)

---

# 4. Public API Contract

## 4.1 Entry Class

```python
CounterfactualExplainer(
    method: Literal[
        "nearest",
        "comte",
        "generative",
        "motif",
        "genetic"
    ],
    model,
    normal_core: np.ndarray,  # shape (K, L, F)
    threshold: Optional[float] = None,
    **method_kwargs
)
```

---

## 4.2 Input Requirements

* Single window only
* Shape `(L, F)`
* Type `np.ndarray`
* No NaN (v1.2)

---

## 4.3 Output Types

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

# 5.1 COMTE — Segment Substitution

## Concept

COMTE replaces only the anomalous segment using a matched normal donor segment from the normal core.

It is:

* Deterministic
* Interpretable
* Data-driven
* Realistic (uses real normal data)

---

## 5.1.1 Pipeline

### Step 1 — Candidate Segment Detection

1. Compute per-timestep error ( e_t )
2. Select top 10% highest-error timesteps
3. Extract largest contiguous block
4. Apply ±2 padding
5. Enforce minimum length = max(5, L//20)

---

### Step 2 — Normal Core Matching

Select donor window:

[
\text{donor} = \arg\min_k |x - x_k^{normal}|^2
]

Matching may use:

* Whole-window MSE
* Segment-only MSE (optional extension)

---

### Step 3 — Segment Substitution

Replace:

[
x_{cf}[s:e] = x_{donor}[s:e]
]

Optional:

* Boundary smoothing
* Blending window edges

---

### Step 4 — Constraint Enforcement

Apply:

* Immutable features
* Value bounds

---

### Step 5 — Validation

Compute new score.

If:

* score <= threshold → valid
* else → try next donor (if enabled)
* else → failure

---

## 5.1.2 Meta Fields

* segment_start
* segment_end
* donor_idx
* score_before
* score_after
* smoothing_used
* attempts

---

# 5.2 Generative — Mask + Infilling

## Concept

Instead of copying from normal data, this method:

* Masks anomalous region
* Uses reconstruction model (or auxiliary infilling engine)
* Generates new values conditioned on surrounding context

This allows more flexible transformations than COMTE.

---

# 5.2.1 Mask Strategy

Mask is determined by:

1. Per-timestep error ranking
2. Contiguous segment extraction
3. Optional multiple mask schedules
4. Immutable features excluded from masking

Mask is binary:

[
M_t =
\begin{cases}
1 & \text{masked (to modify)} \
0 & \text{keep original}
\end{cases}
]

---

# 5.2.2 Infilling Engine

Infilling can be:

* Reconstruction-based (AE reconstruction over masked region)
* Denoising-style infill
* Iterative refinement

Minimum v1.2 requirement:

* Replace masked region with model reconstruction

[
x_{cf} = (1-M) \cdot x + M \cdot \hat{x}
]

---

# 5.2.3 Candidate Selection

Multiple mask strategies may be attempted:

* Small mask
* Expanded mask
* Progressive widening

Best candidate chosen based on:

* Validity (score <= threshold)
* Minimal proximity
* Fewest masked timesteps

---

# 5.2.4 Constraint Handling

After infilling:

* Apply immutable features
* Clip bounds
* Optional smoothing

---

# 5.2.5 Failure Cases

Return failure if:

* No masks found
* All masks exceed threshold
* Over-masking
* Reconstruction collapses (flatline)

---

## 5.2.6 Meta Fields

* mask_size
* mask_start
* mask_end
* attempts
* score_before
* score_after
* best_mask_ratio

---

# 5.3 Comparison: COMTE vs Generative

| Aspect           | COMTE                | Generative                |
| ---------------- | -------------------- | ------------------------- |
| Source of values | Real normal data     | Model-generated           |
| Deterministic    | Yes                  | Possibly stochastic       |
| Realism          | Strong (data-driven) | Depends on model          |
| Flexibility      | Limited to core      | Can create new patterns   |
| Failure mode     | No good donor        | Model reconstruction weak |

---

# 6. Constraints (v1.2)

## 6.1 Immutable Features

```python
immutable_features = [2, 5]
```

These columns remain unchanged.

---

## 6.2 Value Bounds

```python
bounds = {feature_index: (min_val, max_val)}
```

Clipping mandatory.

---

## 6.3 Future Extensions (Not Required in v1.2)

* PCA manifold constraint
* Spectral constraint
* Temporal smoothness regularization

---

# 7. Threshold Estimation

If threshold is None:

1. Compute score for each normal_core window.
2. Set threshold = quantile(scores, 0.95).

---

# 8. Package Structure

```
src/cftsad/
    __init__.py
    api.py
    types.py
    methods/
        nearest.py
        comte.py
        generative.py
        motif.py
        genetic.py
    core/
        scoring.py
        constraints.py
        distances.py
        masking.py
```

---

# 9. Determinism Policy

* nearest → deterministic
* comte → deterministic
* generative → stochastic if sampling used
* genetic → stochastic

All stochastic methods MUST expose:

```python
random_seed
```

Default = 42.

---

# 10. Error Handling

Possible failure reasons:

* invalid_input
* no_valid_cf
* mask_generation_failed
* donor_not_found
* constraint_violation
* reconstruction_failed

---

# 11. Computational Complexity

| Method     | Complexity                    |
| ---------- | ----------------------------- |
| Nearest    | O(KLF)                        |
| COMTE      | O(KLF)                        |
| Generative | O(mask_variants * model_eval) |
| Genetic    | O(pop * gen * model_eval)     |

---

# 12. Usage Example (Generative)

```python
explainer = CounterfactualExplainer(
    method="generative",
    model=model,
    normal_core=core,
    threshold=None,
    max_mask_ratio=0.3,
)

result = explainer.explain(x)
print(result.score_cf)
```

---

# 13. Usage Example (COMTE)

```python
explainer = CounterfactualExplainer(
    method="comte",
    model=model,
    normal_core=core,
    threshold=None,
)

result = explainer.explain(x)
print(result.score_cf)
```

---

# 14. Philosophy

This library unifies:

* Data-driven counterfactuals (COMTE)
* Model-driven counterfactuals (Generative)
* Optimization-driven counterfactuals (Genetic)

All under:

* A single scoring rule
* A single API contract
* Clear validity definition
* Stable package structure

The COMTE method ensures realism through normal-core substitution.
The Generative method allows flexible adaptation beyond available normal samples.
The Genetic method enables continuous optimization when discrete substitution fails.

---

END OF FILE
