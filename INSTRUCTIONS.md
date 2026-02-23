Below is a **production-ready `INSTRUCTIONS.md`** file you can place in the root of your repository.
It defines architecture, API contract, scoring, method behaviour, constraints, and packaging rules for your first public release.

---

# INSTRUCTIONS.md

## Counterfactual Explanations for Time-Series Anomaly Detection (MSE-based Release v1.0)

---

# 1. Project Goal

This package provides **counterfactual explanations** for reconstruction-based time-series anomaly detection models.

Given:

* A trained reconstruction model
* An anomalous window ( x \in \mathbb{R}^{L \times F} )
* A set of normal windows (normal core)

The library generates:

> A minimally modified version of ( x ) such that its reconstruction MSE is below a defined anomaly threshold.

---

# 2. Core Design Principles

### 2.1 Unified Scoring (Mandatory)

All methods MUST use the same anomaly score:

### Window MSE Score

For input window ( x ) and reconstruction ( \hat{x} ):

Per timestep:
[
e_t = \frac{1}{F} \sum_{f=1}^{F} (x_{t,f} - \hat{x}_{t,f})^2
]

Window score:
[
\text{score}(x) = \frac{1}{L} \sum_{t=1}^{L} e_t
]

This score defines:

* Anomaly decision
* Optimization target
* Counterfactual validity condition

---

### 2.2 Valid Counterfactual Definition

A counterfactual is valid if:

```
score(x_cf) <= threshold
```

Where threshold is either:

* Provided explicitly by user
* Or computed from normal core quantile (default 95%)

---

# 3. Public API Contract

The public interface MUST remain stable.

## 3.1 Entry Class

```python
CounterfactualExplainer(
    method: Literal["nearest", "segment", "motif"],
    model: callable or torch module,
    normal_core: np.ndarray,  # shape (K, L, F)
    threshold: Optional[float] = None,
)
```

---

## 3.2 Input Format

* Single window only
* Shape: `(L, F)`
* Type: `numpy.ndarray`
* No NaN allowed (v1)

---

## 3.3 Output Format

Every method MUST return one of:

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

# 4. Implemented Methods (v1)

---

# 4.1 Method 1 — Nearest Neighbour

## Description

Select closest normal window from normal core.

## Algorithm

1. Compute distance between x and each normal window.
2. Select argmin.
3. Apply constraints.
4. Compute score.
5. Return.

## Distance Metric

MSE distance between windows.

## Meta Fields

* donor_idx
* donor_distance
* score_before
* score_after
* runtime_ms

---

# 4.2 Method 2 — Segment Substitution

## Description

Replace only anomalous segment with corresponding part from donor window.

## Segment Detection (Mandatory v1 Rule)

1. Compute per-timestep error ( e_t )
2. Select top 10% timesteps
3. Extract largest contiguous region
4. Apply ±2 padding
5. Enforce minimum length = max(5, L//20)

---

## Pipeline

1. Detect anomalous segment
2. Select nearest donor
3. Replace segment
4. Apply smoothing (optional)
5. Apply constraints
6. Compute score
7. Return

---

## Meta Fields

* segment_start
* segment_end
* donor_idx
* score_before
* score_after
* smoothing_used

---

# 4.3 Method 3 — Motif-Based Substitution

## Description

Replace anomalous segment using motif-level matching instead of full-window donor.

---

## Motif Index Construction

From normal core:

1. Extract sliding subsegments of length m
2. Z-normalize each subsegment
3. Flatten into vector
4. Store in motif index

---

## Motif Retrieval

1. Detect anomalous segment
2. Z-normalize segment
3. Retrieve top-k motifs by Euclidean distance
4. Replace segment
5. Apply constraints
6. Compute score
7. Return best candidate

---

## Meta Fields

* motif_length
* topk_distances
* chosen_motif_source
* score_before
* score_after

---

# 5. Constraints (v1 Minimal Set)

## 5.1 Immutable Features

User may specify:

```python
immutable_features = [2, 5]
```

These columns MUST remain unchanged.

---

## 5.2 Value Clipping (Optional)

User may specify bounds:

```python
bounds = {feature_index: (min_val, max_val)}
```

Counterfactual values MUST be clipped.

---

No PCA, PSD, DTW, or soft constraints in v1.

---

# 6. Threshold Handling

If threshold is None:

1. Compute score for each normal_core window
2. Set:
   [
   threshold = quantile(scores, 0.95)
   ]

---

# 7. Package Structure (Mandatory)

```
src/cftsad/
    __init__.py
    api.py
    types.py
    methods/
        nearest.py
        segment.py
        motif.py
    core/
        scoring.py
        constraints.py
        distances.py
```

Use `src/` layout.

---

# 8. Model Integration Rules

The model may be:

* Callable: `model(x: np.ndarray) -> np.ndarray`
* Or torch module

Internally:

* Convert numpy to torch if needed
* Always return numpy

The public API MUST remain numpy-based.

---

# 9. Performance Guidelines

* Nearest: O(KLF)
* Segment: O(KLF)
* Motif: O(K * motif_count)

Recommended:

* K <= 500
* Window length <= 300 for public demo

---

# 10. Error Handling

Reject if:

* Input shape mismatch
* NaNs present
* Normal core shape inconsistent
* Threshold invalid

Return `CFFailure` with reason codes:

* "invalid_input"
* "no_valid_cf"
* "constraint_violation"
* "segment_detection_failed"

---

# 11. Reproducibility

If randomness used:

* Provide `random_seed` parameter
* Default = 42

---

# 12. Version Policy

v1.0 includes only:

* Nearest
* Segment
* Motif
* MSE scoring
* Minimal constraints

Future versions may add:

* Generative infilling
* Genetic optimization
* PCA / spectral constraints
* Multivariate DTW
* Batch inference

---

# 13. README Example (Mandatory)

Provide minimal example:

```python
import numpy as np
from cftsad import CounterfactualExplainer

model = lambda x: x  # replace with real model
core = np.load("normal_core.npy")
x = np.load("window.npy")

explainer = CounterfactualExplainer(
    method="segment",
    model=model,
    normal_core=core,
    threshold=None,
)

result = explainer.explain(x)

print(result.score_cf)
```

---

# 14. Philosophy

This library prioritizes:

* Stability
* Simplicity
* Reproducibility
* Clear method separation
* Consistent scoring

Heavy optimization and advanced constraints are intentionally excluded from v1 to ensure reliability and usability.

---

END OF FILE
