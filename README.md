# cftsad

Counterfactual explanations for reconstruction-based time-series anomaly detection.

## Installation

Install from PyPI (after first release):

```bash
pip install cftsad
```

Install directly from source:

```bash
pip install .
```

Editable install (for development):

```bash
pip install -e .
```

Install from GitHub:

```bash
pip install "git+https://github.com/<org-or-user>/cftsad.git"
```

## Minimal example

```python
import numpy as np
from cftsad import CounterfactualExplainer

model = lambda x: x  # replace with your reconstruction model
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

if hasattr(result, "score_cf"):
    print(result.score_cf)
else:
    print(result.reason, result.message)
```

## Method APIs

All methods use the same base API:

```python
from cftsad import CounterfactualExplainer

explainer = CounterfactualExplainer(
    method="nearest",  # "nearest" | "segment" | "motif" | "genetic"
    model=model,
    normal_core=core,  # shape: (K, L, F)
    threshold=None,    # if None, computed from normal_core (q=0.95)
    immutable_features=(0,),              # optional
    bounds={1: (-3.0, 3.0)},              # optional
    random_seed=42,                        # optional
)
result = explainer.explain(x)  # x shape: (L, F)
```

### `nearest`

No method-specific kwargs. Uses the closest donor window from `normal_core`.

```python
explainer = CounterfactualExplainer(
    method="nearest",
    model=model,
    normal_core=core,
    threshold=None,
    immutable_features=(0,),
    bounds={1: (-3.0, 3.0)},
)
```

### `segment`

Method-specific kwarg:
- `segment_smoothing: bool = False`

```python
explainer = CounterfactualExplainer(
    method="segment",
    model=model,
    normal_core=core,
    threshold=None,
    segment_smoothing=True,
)
```

### `motif`

Method-specific kwarg:
- `motif_top_k: int = 5`

```python
explainer = CounterfactualExplainer(
    method="motif",
    model=model,
    normal_core=core,
    threshold=None,
    motif_top_k=10,
)
```

### `genetic`

Method-specific kwargs:
- `population_size: int = 100`
- `n_generations: int = 50`
- `crossover_rate: float = 0.9`
- `mutation_rate: float = 0.1`
- `mutation_sigma: float = 0.05`
- `use_smoothness_objective: bool = False`

```python
explainer = CounterfactualExplainer(
    method="genetic",
    model=model,
    normal_core=core,
    threshold=None,
    population_size=100,
    n_generations=50,
    crossover_rate=0.9,
    mutation_rate=0.1,
    mutation_sigma=0.05,
    use_smoothness_objective=False,
)
```

## Return values

`explain(x)` returns either:
- `CFResult` on success
- `CFFailure` on failure

### `CFResult`

Fields:
- `x_cf: np.ndarray`  
  Counterfactual window with shape `(L, F)`.
- `score_cf: float`  
  Reconstruction score of `x_cf` (lower is better).
- `meta: dict`  
  Method-specific diagnostics (for example `donor_idx`, `segment_start`, `runtime_ms`, `warning`).

### `CFFailure`

Fields:
- `reason: str`  
  Machine-readable failure code.
- `message: str`  
  Human-readable summary of what failed.
- `diagnostics: dict`  
  Extra debugging information (threshold, scores, indices, exception type, etc.).

Common `reason` values include:
- `invalid_input`
- `segment_detection_failed`
- `no_valid_cf`
- `optimization_failed`

### Example: handling return type

```python
from cftsad import CounterfactualExplainer, CFResult, CFFailure

result = explainer.explain(x)

if isinstance(result, CFResult):
    print("success:", result.score_cf, result.x_cf.shape)
    print("meta keys:", result.meta.keys())
elif isinstance(result, CFFailure):
    print("failed:", result.reason)
    print(result.message)
    print("diagnostics:", result.diagnostics)
```
