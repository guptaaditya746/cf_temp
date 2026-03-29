# cftsad

Counterfactual explanations for reconstruction-based time-series anomaly detection.

## Installation

```bash
pip install cftsad
```

From source:

```bash
pip install .
```

Editable dev install:

```bash
pip install -e .
```

## Branch Workflow

Active development now happens on the `genetic` branch.

```bash
git switch genetic
git pull
git push -u origin genetic
```

`main` is kept as the stable/reference branch until the remote default branch is changed.

## Running the Pipeline

The repo now supports a staged pipeline layout:

- `preprocessing.py`
- `training.py`
- `evaluation_model.py`
- `generating_counterfactual.py`
- `evaluation_counterfactual.py`

Main settings live in [configs/defaults.py](/home/gupt_ad/part_2_local/push_repos_only/cf_temp/configs/defaults.py).

### Prerequisites

Install the runtime dependencies used by the pipeline example:

```bash
pip install -e .
pip install torch pytorch-lightning pandas scikit-learn matplotlib seaborn
```

### End-to-End Run

Run the full pipeline step by step using one shared run directory.

```bash
python preprocessing.py
python training.py --run-dir results/run_YYYYMMDD-HHMMSS
python evaluation_model.py --run-dir results/run_YYYYMMDD-HHMMSS
python generating_counterfactual.py --run-dir results/run_YYYYMMDD-HHMMSS
python evaluation_counterfactual.py --run-dir results/run_YYYYMMDD-HHMMSS
```

The first command creates a new run directory under `results/`. Use that same path for all later stages.

### Single-Script Run

To run the complete flow in one go:

```bash
python example.py
```

### Stage Outputs

- preprocessing writes `X_train.npy`, `X_val.npy`, `X_calib.npy`, `X_test.npy`
- training writes checkpoints and `training_metadata.json`
- model evaluation writes files under `results/.../evaluation/`
- counterfactual generation writes `counterfactual_log.csv` and `cf_arrays/`
- counterfactual evaluation writes `counterfactual_summary.csv`

### Customization

- change dataset paths, split ratios, model choice, and method defaults in [configs/defaults.py](/home/gupt_ad/part_2_local/push_repos_only/cf_temp/configs/defaults.py)
- add new black-box models in [models/autoencoder.py](/home/gupt_ad/part_2_local/push_repos_only/cf_temp/models/autoencoder.py) or another file under `models/`, then register them in `MODEL_REGISTRY`
- add new counterfactual pipelines under [counterfactual_methods/pipeline.py](/home/gupt_ad/part_2_local/push_repos_only/cf_temp/counterfactual_methods/pipeline.py)

## Quick Start

```python
import numpy as np
from cftsad import CounterfactualExplainer, CFResult, CFFailure

model = lambda x: x  # replace with your reconstruction model
core = np.load("normal_core.npy")  # (K, L, F)
x = np.load("window.npy")          # (L, F)

def score_fn(window: np.ndarray) -> float:
    recon = np.asarray(model(window))
    return float(np.mean((window - recon) ** 2))

explainer = CounterfactualExplainer(
    method="genetic",
    model=model,
    normal_core=core,
    score_fn=score_fn,  # required: use the same scoring function as evaluation
    threshold=None,
    population_size=100,
    n_generations=50,
)

result = explainer.explain(x)
if isinstance(result, CFResult):
    print("score:", result.score_cf)
else:
    print(result.reason, result.message)
```

## Model API Contract

`model` must return a **reconstruction window**, not an anomaly score.

- input accepted by `cftsad`: `(L, F)` (and `(1, L, F)` fallback is supported)
- output expected: same shape as input window (`(L, F)` or squeezable `(1, L, F)`)

`score_fn` is required so counterfactual generation uses the exact same window-level scoring rule as evaluation.

### PyTorch usage (recommended)

For `torch.nn.Module`, `cftsad` automatically:
- uses `eval()`
- runs in `torch.no_grad()`
- casts to `float32`
- adds batch dim for `(L, F)` input when needed

#### Option 1: pass `nn.Module` directly

```python
import torch
from cftsad import CounterfactualExplainer

class ReconModel(torch.nn.Module):
    def forward(self, x):  # x: (B, L, F)
        return x           # replace with your model

model = ReconModel()
explainer = CounterfactualExplainer(
    method="nearest",
    model=model,
    normal_core=core,
    score_fn=score_fn,
    threshold=None,
)
```

#### Option 2: Lightning/custom wrapper callable

Use this when your module returns extra outputs such as `(recon, loss)`.

```python
import numpy as np
import torch

def make_recon_callable(pl_module):
    pl_module.eval()

    def recon_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.as_tensor(x_np, dtype=torch.float32, device=pl_module.device)
            if x_t.ndim == 2:
                x_t = x_t.unsqueeze(0)  # (1, L, F)

            out = pl_module(x_t)
            if isinstance(out, (tuple, list)):
                out = out[0]

            out_np = out.detach().cpu().numpy()
            if out_np.ndim == 3 and out_np.shape[0] == 1:
                out_np = out_np[0]
            return out_np

    return recon_fn
```

## Input Requirements

- `normal_core`: `np.ndarray` shape `(K, L, F)`
- `x` passed to `explain(x)`: shape `(L, F)`
- `x` and `normal_core` must share `(L, F)`
- no NaNs in `normal_core` or `x`

## CounterfactualExplainer API

### Base constructor

```python
CounterfactualExplainer(
    method="nearest",  # "nearest" | "segment" | "motif" | "genetic"
    model=model,
    normal_core=core,
    score_fn=score_fn,              # required: shared evaluation scoring function
    threshold=None,
    immutable_features=(0,),        # optional
    bounds={1: (-3.0, 3.0)},        # optional
    random_seed=42,                 # optional
)
```

### Common arguments

- `method`
- `model`
- `normal_core`
- `score_fn`: required callable used for threshold/core building and counterfactual scoring
- `threshold`: if `None`, estimated from normal core
- `immutable_features`
- `bounds`
- `random_seed`

### Normal-core builder arguments

- `normal_core_filter_factor: float = 1.0`
- `normal_core_threshold_quantile: float = 0.95`
- `normal_core_max_size: Optional[int] = None`
- `normal_core_embedding_dim: Optional[int] = 32`
- `normal_core_use_diversity_sampling: bool = True`

Build artifacts exposed on explainer:
- `core_index`
- `core_embeddings`
- `core_reduced_embeddings`
- `core_pca_components`
- `core_pca_mean`
- `core_scores`
- `core_scores_all`
- `core_build_info`

### Constraint system (v2)

Supported across nearest/segment/motif/genetic:
- `use_constraints_v2: bool = True`
- `max_delta_per_step: Optional[float] = None`
- `relational_linear: Optional[dict[str, tuple[int, int, float]]] = None`

`relational_linear` format:
- `{"rule_name": (lhs_feature_idx, rhs_feature_idx, min_ratio)}`
- interpreted as soft rule: `x[:, lhs] >= min_ratio * x[:, rhs]`

## Method-Specific Arguments

### `nearest`

- `nearest_top_k: int = 10`
- `nearest_alpha_steps: int = 11`
- `nearest_donor_filter_factor: float = 1.0`
- `nearest_use_weighted_distance: bool = True`

### `segment`

- `segment_smoothing: bool = False`
- `segment_n_candidates: int = 4`
- `segment_top_k_donors: int = 8`
- `segment_context_width: int = 2`
- `segment_crossfade_width: int = 3`

### `motif`

- `motif_top_k: int = 5`
- `motif_n_segments: int = 4`
- `motif_length_factors: tuple[float, ...] = (0.75, 1.0, 1.25)`
- `motif_context_weight: float = 0.2`
- `motif_use_affine_fit: bool = True`

### `genetic`

- `population_size: int = 100`
- `n_generations: int = 50`
- `crossover_rate: float = 0.9`
- `mutation_rate: float = 0.1`
- `mutation_sigma: float = 0.05`
- `use_smoothness_objective: bool = False`
- `use_plausibility_objective: bool = True`
- `structured_mutation_weight: float = 0.35`
- `validity_margin: float = 0.0`
- `top_m_solutions: int = 5`
- `early_stop_patience: int = 15`

## Fallback Chain

If primary method fails with:
- `no_valid_cf`
- `segment_detection_failed`
- `optimization_failed`

then `explain(...)` can try other methods automatically.

Arguments:
- `enable_fallback_chain: bool = True`
- `fallback_methods: tuple[str, ...] = ("segment", "motif", "nearest")`
- `fallback_retry_budget: int = 3`

Example:

```python
explainer = CounterfactualExplainer(
    method="segment",
    model=model,
    normal_core=core,
    score_fn=score_fn,
    threshold=0.1,
    enable_fallback_chain=True,
    fallback_methods=("motif", "nearest", "genetic"),
    fallback_retry_budget=3,
)
```

## Return Types

`explain(x)` returns either:
- `CFResult`
- `CFFailure`

### `CFResult`

- `x_cf: np.ndarray` (shape `(L, F)`)
- `score_cf: float`
- `meta: dict`

`meta` may include:
- method diagnostics (donor/segment/motif source, runtime)
- objective summaries (`genetic`)
- constraints diagnostics (`constraint_violation`, `constraint_breakdown`)
- explainability artifacts:
  - `changed_features`
  - `changed_timesteps`
  - `top_changed_features`
  - `top_changed_timesteps`
  - `edit_mask`
  - `delta_l1`, `delta_l2`
- fallback metadata (`fallback_used`, `fallback_from`, `fallback_method`)

### `CFFailure`

- `reason: str`
- `message: str`
- `diagnostics: dict`

Common `reason` values:
- `invalid_input`
- `segment_detection_failed`
- `no_valid_cf`
- `optimization_failed`

## Core Utilities

`cftsad.core` provides reusable helpers:

- candidate ranking/orchestration:
  - `compute_candidate_metrics`
  - `rank_candidates`
  - `deduplicate_candidates`
  - `prune_candidates`
  - `evaluate_candidate_pool`
- attribution:
  - `attribution_from_delta`
  - `attribution_from_reconstruction_error`
- normal-core indexing:
  - `transform_embedding`
  - `query_core_index`
- persistence:
  - `save_core_artifacts`
  - `load_core_artifacts`

### Save/Load core artifacts from explainer

```python
explainer.save_core("/tmp/cftsad_core.npz")
loaded = CounterfactualExplainer.load_core("/tmp/cftsad_core.npz")
print(loaded.keys())
```




### legacy
1. genetic : mostly its failing , so could be worked on latter on, but for now the probalabe cause its trying ot optiomize mostly validity
2.
