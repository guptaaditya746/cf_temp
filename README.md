# cftsad

Counterfactual explanations for reconstruction-based time-series anomaly detection.

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
