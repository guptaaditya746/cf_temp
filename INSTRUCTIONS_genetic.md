
Excellent — this is the right move.

If you want the **genetic method to look research-level and theoretically grounded**, it must:

* Be framed as a **constrained multi-objective optimization problem**
* Clearly define **feasible region geometry**
* Separate **hard constraints vs soft regularizers**
* Define a principled **solution selection rule**
* Include convergence + stability analysis hooks

Below is your updated research-grade file:

---

# INSTRUCTIONS_genetic.md

## Theoretical Genetic Counterfactual Optimization

### NSGA-II Multi-Objective Formulation for Time-Series Anomaly Detection

Version 2.0 (Research-Oriented)

---

# 1. Formal Problem Definition

Let:

* ( x \in \mathbb{R}^{L \times F} ) be an anomalous window
* ( f(x) ) be reconstruction MSE
* ( \tau ) be anomaly threshold
* ( \mathcal{C} \subset \mathbb{R}^{L \times F} ) be constraint manifold

We define counterfactual generation as:

[
\min_{x'} \quad \mathbf{F}(x') =
\begin{bmatrix}
f(x') \
d(x, x') \
s(x, x')
\end{bmatrix}
\quad
\text{s.t.} \quad
x' \in \mathcal{C}
]

Where:

* ( f(x') ) → validity objective
* ( d(x,x') ) → proximity objective
* ( s(x,x') ) → sparsity objective
* ( \mathcal{C} ) → feasible region

This defines a **multi-objective constrained optimization problem**.

---

# 2. Objective Definitions

---

## 2.1 Objective 1 — Validity (Anomaly Score)

[
f_1(x') = \text{score}(x') = \frac{1}{L} \sum_{t=1}^{L} \frac{1}{F} \sum_{f=1}^{F} (x'*{t,f} - \hat{x'}*{t,f})^2
]

Goal:
[
f_1(x') \le \tau
]

This objective defines the anomaly boundary.

---

## 2.2 Objective 2 — Proximity

[
f_2(x') = |x' - x|_2^2
]

This ensures minimal modification.

---

## 2.3 Objective 3 — Sparsity

[
f_3(x') = \frac{1}{L} \sum_{t=1}^{L} \mathbf{1}\left( |x'_t - x_t|_2 > \epsilon \right)
]

Encourages few modified timesteps.

---

## 2.4 Optional Objective 4 — Temporal Smoothness

[
f_4(x') = \sum_{t=2}^{L} |x'*t - x'*{t-1}|^2
]

Controls unrealistic oscillations.

---

# 3. Feasible Region ( \mathcal{C} )

The feasible region is defined as:

[
\mathcal{C} =
\mathcal{C}*{immutable}
\cap
\mathcal{C}*{bounds}
\cap
\mathcal{C}_{manifold}
]

---

## 3.1 Immutable Constraint

For features ( j \in I ):

[
x'*{:,j} = x*{:,j}
]

Hard constraint.

---

## 3.2 Value Bounds

For each feature ( j ):

[
l_j \le x'_{t,j} \le u_j
]

Hard constraint.

---

## 3.3 Manifold Constraint (Research-Level Extension)

Optional PCA-based coupling:

Let ( \Phi ) be PCA basis learned from normal core.

Then:

[
x' \approx \Phi \Phi^T x'
]

Soft constraint via penalty:

[
\lambda | x' - \Phi \Phi^T x' |^2
]

---

# 4. NSGA-II Optimization Framework

We use NSGA-II because:

* Counterfactual objectives are inherently conflicting
* No scalarization assumption required
* Produces Pareto frontier

---

# 5. Representation

Each individual encodes:

[
x' \in \mathbb{R}^{L \times F}
]

Optimization occurs in continuous space.

---

# 6. Initialization Strategy (Theoretical Improvement)

Instead of random initialization:

[
x'_0 = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
]

Additionally:

* Include donor-based initialization
* Include reconstruction projection

This improves convergence guarantees.

---

# 7. Constraint Handling Strategy

We adopt feasibility-first ranking:

1. Feasible solutions dominate infeasible ones
2. Among infeasible:

   * Rank by total violation magnitude

Violation magnitude:

[
CV(x') = \sum_k \max(0, g_k(x'))
]

---

# 8. Pareto Selection Rule (Theoretical)

After optimization:

Select:

[
x^* =
\arg\min_{x' \in \mathcal{P}}
|x' - x|
\quad
\text{s.t.}
\quad
f_1(x') \le \tau
]

If none satisfy validity:

Return best compromise with diagnostic flag.

---

# 9. Convergence Considerations

Assuming:

* Continuous reconstruction function
* Bounded search space
* Non-zero mutation probability

NSGA-II converges (in probability) toward Pareto-optimal set.

We do NOT guarantee global optimality.

---

# 10. Counterfactual Difficulty Score (Research Addition)

Define:

[
D(x) = \min_{x'} |x' - x|
\quad
\text{s.t.}
\quad
f_1(x') \le \tau
]

We estimate ( D(x) ) from Pareto front.

This measures anomaly severity.

---

# 11. Stability Metric (Research Addition)

For perturbation ( \epsilon ):

[
S(x) =
| CF(x) - CF(x + \epsilon) |
]

Low value implies stable explanation.

---

# 12. Hyperparameters (Public)

```python
population_size = 100
n_generations = 100
crossover_rate = 0.9
mutation_rate = 0.1
mutation_sigma = 0.05
random_seed = 42
```

---

# 13. Computational Complexity

[
O(population_size \times generations \times model_evaluation)
]

Dominated by reconstruction model cost.

---

# 14. Output Metadata (Extended)

```python
meta = {
    "pareto_size": int,
    "best_objectives": [f1, f2, f3],
    "difficulty_score": float,
    "constraint_violation": float,
    "generations": int,
    "population_size": int,
    "runtime_ms": float,
}
```

---

# 15. Research Positioning

This method provides:

* Continuous counterfactual optimization
* Multi-objective trade-off analysis
* Geometric view of feasible region
* Quantification of anomaly difficulty
* Stability analysis capability

It generalizes deterministic substitution methods and enables theoretical study of:

* Counterfactual manifold geometry
* Trade-off structure
* Optimization landscape smoothness

---

# 16. Limitations

* Computationally expensive
* Sensitive to hyperparameters
* No formal global optimality guarantee
* Requires careful constraint tuning

---

# 17. Future Research Extensions

* Differential evolution variant
* Hybrid gradient-assisted mutation
* Trust-region evolutionary search
* Diffusion-initialized population
* Constrained Bayesian optimization alternative

---

END OF FILE

---

If you want next, I can:

* Turn this into a **paper-ready methodology section**
* Or mathematically refine the convergence analysis
* Or convert this into a NeurIPS/ICLR-style method section draft
