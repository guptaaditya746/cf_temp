# optimizers.py
# Part 4: Optimization Strategy (library-first)
#
# Provides:
# - CMA-ES optimizer wrapper (pycma) for scalar loss
# - NSGA-II optimizer wrapper (pymoo) for multi-objective Pareto search
#
# Integrates with:
# - Part 1: LatentCFProblem (latent_problem.py)
# - Part 2: DecodedConstraintEvaluator (decoded_constraints.py)
# - Part 3: ScalarLatentCFObjective / MultiObjectiveLatentCF (objectives.py)
#
# HARD RULES:
# - No gradients
# - Validity is ALWAYS evaluated on decoded score
# - Hard decoded constraints reject candidates immediately
# - Latent projection (bounds/mask/locality) applied to every candidate
#
# External-first:
# - pycma (CMA-ES)
# - pymoo (NSGA-II)
# - numpy, torch

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Local imports (your project modules)
# from latent_problem import LatentCFProblem
# from decoded_constraints import DecodedConstraintEvaluator, ConstraintEvaluationResult
# from objectives import ScalarLatentCFObjective, MultiObjectiveLatentCF, MultiObjectiveValues
#
# Note: Keep these as relative imports if packaging:
# from .latent_problem import LatentCFProblem
# from .decoded_constraints import DecodedConstraintEvaluator, ConstraintEvaluationResult
# from .objectives import ScalarLatentCFObjective, MultiObjectiveLatentCF, MultiObjectiveValues

Tensor = torch.Tensor


# -------------------------
# Shared evaluation helpers
# -------------------------


@dataclass
class CandidateEval:
    z_cf: Tensor
    score: float
    hard_ok: bool
    constraint_penalty: float
    constraint_metrics: Dict[str, float]
    loss: Optional[float] = None
    objectives: Optional[np.ndarray] = None
    failure_reason: Optional[str] = None


def _to_numpy(z: Tensor) -> np.ndarray:
    return z.detach().cpu().numpy().astype(np.float64, copy=False)


def _from_numpy(z_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.from_numpy(np.asarray(z_np, dtype=np.float32)).to(
        device=device, dtype=dtype
    )


def _ensure_py_cma() -> Any:
    try:
        import cma  # pycma package name is "cma"

        return cma
    except Exception as e:
        raise RuntimeError(
            "CMA-ES requested but pycma is not installed. Install with: pip install cma"
        ) from e


def _ensure_pymoo() -> Any:
    try:
        import pymoo

        return pymoo
    except Exception as e:
        raise RuntimeError(
            "NSGA-II requested but pymoo is not installed. Install with: pip install pymoo"
        ) from e


# -------------------------
# CMA-ES (scalar objective)
# -------------------------


@dataclass
class CMAESConfig:
    sigma0: float = 0.5
    popsize: Optional[int] = None
    max_evals: int = 2000
    seed: Optional[int] = None

    # Candidate handling
    infeasible_loss: float = 1e9  # loss assigned to hard-infeasible decoded candidates

    # Optional early stop
    stop_on_first_valid: bool = False

    def validate(self) -> None:
        if self.sigma0 <= 0:
            raise ValueError("sigma0 must be > 0")
        if self.max_evals <= 0:
            raise ValueError("max_evals must be > 0")
        if self.popsize is not None and self.popsize <= 2:
            raise ValueError("popsize must be > 2 if provided")
        if self.infeasible_loss <= 0:
            raise ValueError("infeasible_loss must be > 0")


class CMAESLatentOptimizer:
    """
    CMA-ES optimizer wrapper for scalar-loss latent CF.
    """

    def __init__(
        self,
        *,
        config: CMAESConfig,
    ):
        config.validate()
        self.cfg = config

    def run(
        self,
        *,
        problem: Any,  # LatentCFProblem
        constraint_eval: Any,  # DecodedConstraintEvaluator
        scalar_objective: Any,  # ScalarLatentCFObjective
        x_orig: Tensor,
    ) -> Dict[str, Any]:
        """
        Returns dict with:
          - best: CandidateEval or None
          - history: list[CandidateEval] (evaluated)
          - n_evals: int
          - optimizer: str
        """
        if problem.z is None or problem.latent_dim is None:
            raise RuntimeError(
                "problem must be built via problem.build(x) before optimization"
            )

        cma = _ensure_py_cma()

        z0 = problem.z
        D = problem.latent_dim
        device, dtype = z0.device, z0.dtype

        # CMA-ES works on numpy vectors.
        x0 = _to_numpy(z0)

        opts: Dict[str, Any] = {}
        if self.cfg.popsize is not None:
            opts["popsize"] = int(self.cfg.popsize)
        if self.cfg.seed is not None:
            opts["seed"] = int(self.cfg.seed)
        opts["maxfevals"] = int(self.cfg.max_evals)
        opts["verb_disp"] = 0
        opts["verbose"] = -9  # silence

        es = cma.CMAEvolutionStrategy(x0, float(self.cfg.sigma0), opts)

        history: List[CandidateEval] = []
        best: Optional[CandidateEval] = None
        n_evals = 0

        validity_target = problem.validity_target()

        def eval_one(z_np: np.ndarray) -> CandidateEval:
            nonlocal n_evals, best
            n_evals += 1

            z_t = _from_numpy(z_np, device=device, dtype=dtype)
            z_t = problem.apply_search_space(z_t)  # bounds/mask/locality projection

            # Decode + hard/soft constraints
            x_cf = problem.decode(z_t)
            cres = constraint_eval.evaluate(x_cf, x_orig=x_orig)

            if not cres.hard_ok:
                ce = CandidateEval(
                    z_cf=z_t,
                    score=float("inf"),
                    hard_ok=False,
                    constraint_penalty=float("inf"),
                    constraint_metrics=cres.metrics,
                    loss=self.cfg.infeasible_loss,
                    failure_reason=cres.failure_reason,
                )
                return ce

            # Score
            s = problem.score_fn(x_cf)
            s = float(s)

            # Scalar objective
            loss, diag = scalar_objective(
                z_cf=z_t,
                z=z0,
                score_cf=s,
                validity_target=validity_target,
                constraint_penalty=float(cres.soft_penalty),
                normalcore_z=problem.normalcore_z,
            )

            ce = CandidateEval(
                z_cf=z_t,
                score=s,
                hard_ok=True,
                constraint_penalty=float(cres.soft_penalty),
                constraint_metrics=cres.metrics,
                loss=float(loss),
                failure_reason=None,
            )

            # Track best
            if best is None or (ce.loss is not None and ce.loss < best.loss):
                best = ce

            return ce

        # Main loop
        while not es.stop():
            Z = es.ask()
            vals = []
            batch_evals = []
            for z_np in Z:
                ce = eval_one(np.asarray(z_np, dtype=np.float64))
                batch_evals.append(ce)
                history.append(ce)
                vals.append(
                    float(ce.loss if ce.loss is not None else self.cfg.infeasible_loss)
                )

            es.tell(Z, vals)

            if self.cfg.stop_on_first_valid:
                # "valid" means decoded score meets threshold margin AND hard constraints ok
                for ce in batch_evals:
                    if ce.hard_ok and ce.score <= validity_target:
                        best = ce
                        es.stop()  # request stop
                        break

            if n_evals >= self.cfg.max_evals:
                break

        return {
            "best": best,
            "history": history,
            "n_evals": n_evals,
            "optimizer": "CMA-ES(pycma)",
        }


# -------------------------
# NSGA-II (multi-objective)
# -------------------------


@dataclass
class NSGA2Config:
    pop_size: int = 64
    n_gen: int = 50
    seed: Optional[int] = None

    # Candidate handling
    infeasible_obj: float = 1e6  # objective value assigned to infeasible candidates

    # If you want to treat validity as epsilon-constraint rather than objective:
    use_validity_epsilon_constraint: bool = False
    validity_eps: float = 0.0  # allow tiny excess; normally 0

    def validate(self) -> None:
        if self.pop_size <= 4:
            raise ValueError("pop_size must be > 4")
        if self.n_gen <= 0:
            raise ValueError("n_gen must be > 0")
        if self.infeasible_obj <= 0:
            raise ValueError("infeasible_obj must be > 0")
        if self.validity_eps < 0:
            raise ValueError("validity_eps must be >= 0")


class NSGA2LatentOptimizer:
    """
    NSGA-II optimizer wrapper using pymoo.

    IMPORTANT:
    - We still evaluate decoded constraints + score as black box
    - We build objective vectors per candidate
    - Infeasible candidates get large objective penalties
    """

    def __init__(self, *, config: NSGA2Config):
        config.validate()
        self.cfg = config

    def run(
        self,
        *,
        problem: Any,  # LatentCFProblem
        constraint_eval: Any,  # DecodedConstraintEvaluator
        multi_objective: Any,  # MultiObjectiveLatentCF
        x_orig: Tensor,
    ) -> Dict[str, Any]:
        """
        Returns dict with:
          - pareto: list[CandidateEval]
          - history: list[CandidateEval] (evaluated; may be large)
          - n_evals: int
          - optimizer: str
        """
        _ensure_pymoo()
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.optimize import minimize

        if problem.z is None or problem.latent_dim is None:
            raise RuntimeError(
                "problem must be built via problem.build(x) before optimization"
            )

        z0 = problem.z
        D = problem.latent_dim
        device, dtype = z0.device, z0.dtype
        validity_target = problem.validity_target()

        # Bounds for pymoo: if not provided, use wide defaults (still projected by apply_search_space)
        if problem.search_space.bounds is not None:
            b = problem.search_space.bounds.detach().cpu().numpy().astype(np.float64)
            xl = b[:, 0]
            xu = b[:, 1]
        else:
            # Safe fallback; true restriction is still apply_search_space (locality etc.)
            xl = -10.0 * np.ones(D, dtype=np.float64)
            xu = +10.0 * np.ones(D, dtype=np.float64)

        history: List[CandidateEval] = []
        eval_counter = {"n": 0}

        class LatentNSGAProblem(PymooProblem):
            def __init__(self):
                # 4 objectives: latent_dist, validity_excess, constraint_penalty, normalcore_dist
                super().__init__(
                    n_var=D,
                    n_obj=4,
                    n_constr=0,
                    xl=xl,
                    xu=xu,
                    elementwise_evaluation=False,
                )

            def _evaluate(self, X, out, *args, **kwargs):
                # X: (N,D) numpy
                N = X.shape[0]
                F = np.zeros((N, 4), dtype=np.float64)

                for i in range(N):
                    eval_counter["n"] += 1

                    z_t = _from_numpy(X[i], device=device, dtype=dtype)
                    z_t = problem.apply_search_space(z_t)

                    try:
                        x_cf = problem.decode(z_t)
                        cres = constraint_eval.evaluate(x_cf, x_orig=x_orig)
                        if not cres.hard_ok:
                            ce = CandidateEval(
                                z_cf=z_t,
                                score=float("inf"),
                                hard_ok=False,
                                constraint_penalty=float("inf"),
                                constraint_metrics=cres.metrics,
                                objectives=np.array(
                                    [self_outer.cfg.infeasible_obj] * 4,
                                    dtype=np.float64,
                                ),
                                failure_reason=cres.failure_reason,
                            )
                            history.append(ce)
                            F[i, :] = ce.objectives
                            continue

                        s = float(problem.score_fn(x_cf))

                        mo_vals = multi_objective(
                            z_cf=z_t,
                            z=z0,
                            score_cf=s,
                            validity_target=validity_target,
                            constraint_penalty=float(cres.soft_penalty),
                            normalcore_z=problem.normalcore_z,
                        )
                        obj = mo_vals.as_array()

                        # Optional: treat validity as epsilon-constraint by hard-penalizing excess
                        if self_outer.cfg.use_validity_epsilon_constraint:
                            if mo_vals.validity_excess > self_outer.cfg.validity_eps:
                                obj = obj.copy()
                                obj[1] = (
                                    self_outer.cfg.infeasible_obj
                                )  # punish validity objective

                        ce = CandidateEval(
                            z_cf=z_t,
                            score=s,
                            hard_ok=True,
                            constraint_penalty=float(cres.soft_penalty),
                            constraint_metrics=cres.metrics,
                            objectives=obj,
                            failure_reason=None,
                        )
                        history.append(ce)
                        F[i, :] = obj

                    except Exception as e:
                        # Any evaluation crash => treat as infeasible
                        obj = np.array(
                            [self_outer.cfg.infeasible_obj] * 4, dtype=np.float64
                        )
                        ce = CandidateEval(
                            z_cf=z_t,
                            score=float("inf"),
                            hard_ok=False,
                            constraint_penalty=float("inf"),
                            constraint_metrics={},
                            objectives=obj,
                            failure_reason=f"exception:{type(e).__name__}",
                        )
                        history.append(ce)
                        F[i, :] = obj

                out["F"] = F

        # Trick to access self.cfg from inner class
        self_outer = self

        algo = NSGA2(pop_size=int(self.cfg.pop_size))
        res = minimize(
            LatentNSGAProblem(),
            algo,
            ("n_gen", int(self.cfg.n_gen)),
            seed=None if self.cfg.seed is None else int(self.cfg.seed),
            verbose=False,
        )

        # Extract Pareto set (decoded again for safety, but we reuse stored if possible)
        pareto: List[CandidateEval] = []
        if res.X is not None:
            for i in range(res.X.shape[0]):
                z_t = _from_numpy(res.X[i], device=device, dtype=dtype)
                z_t = problem.apply_search_space(z_t)
                x_cf = problem.decode(z_t)
                cres = constraint_eval.evaluate(x_cf, x_orig=x_orig)
                if not cres.hard_ok:
                    continue
                s = float(problem.score_fn(x_cf))
                mo_vals = multi_objective(
                    z_cf=z_t,
                    z=z0,
                    score_cf=s,
                    validity_target=validity_target,
                    constraint_penalty=float(cres.soft_penalty),
                    normalcore_z=problem.normalcore_z,
                )
                ce = CandidateEval(
                    z_cf=z_t,
                    score=s,
                    hard_ok=True,
                    constraint_penalty=float(cres.soft_penalty),
                    constraint_metrics=cres.metrics,
                    objectives=mo_vals.as_array(),
                )
                pareto.append(ce)

        return {
            "pareto": pareto,
            "history": history,
            "n_evals": int(eval_counter["n"]),
            "optimizer": "NSGA-II(pymoo)",
        }


# -------------------------
# Minimal smoke test (no external libs required to import)
# -------------------------
if __name__ == "__main__":
    print("This module defines Part 4 optimizers (CMA-ES and NSGA-II).")
    print("To run CMA-ES, install: pip install cma")
    print("To run NSGA-II, install: pip install pymoo")
