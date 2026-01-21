from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.termination import Termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


@dataclass(frozen=True)
class NSGA2Config:
    pop_size: int = 80
    n_gen: int = 60
    seed: int = 7
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    eliminate_duplicates: bool = True


@dataclass
class NSGA2Result:
    X: np.ndarray
    F: np.ndarray
    G: Optional[np.ndarray]
    feasible: Optional[np.ndarray]
    best_idx: Optional[int]
    meta: Dict[str, Any]


class _CallbackProblem(Problem):
    """
    A generic multi-objective problem that delegates evaluation to a user callback.

    The decision variables are continuous floats in [xl, xu].
    The callback must map each row of X to (F, G, meta) where:
      - F: (n_obj,) float array (objectives, minimize)
      - G: (n_constr,) float array where G <= 0 means constraint satisfied (optional)
      - meta: dict (optional, ignored by optimizer)
    """

    def __init__(
        self,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
        n_obj: int,
        n_constr: int,
        eval_fn: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
    ):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl.astype(float),
            xu=xu.astype(float),
            elementwise_evaluation=False,
        )
        self._eval_fn = eval_fn

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = self._eval_fn(X)
        out["F"] = F
        if self.n_constr > 0:
            if G is None:
                raise ValueError("Problem expects constraints (n_constr > 0) but eval_fn returned None for G.")
                raise ValueError(
                    "Problem expects constraints (n_constr > 0) but eval_fn returned None for G."
                )
            out["G"] = G


class NSGA2Optimizer:
    """
    Thin wrapper around pymoo NSGA-II.

    This module is intentionally domain-agnostic. It assumes:
      - Decision variables are real-valued in [xl, xu]
      - Objectives are to be minimized
      - Constraints (if any) are in pymoo form: G <= 0 is feasible

    The domain-specific module (e.g., cf_problem_segment.py) should:
      - define encoding/decoding between genome and X (continuous vector)
      - define eval_fn(X) -> (F, G)
      - define how to pick the final solution from result.X/result.F with any extra metadata
    """

    def __init__(self, config: NSGA2Config):
        self.config = config

    def run(
        self,
        *,
        n_var: int,
        xl: Union[Sequence[float], np.ndarray],
        xu: Union[Sequence[float], np.ndarray],
        n_obj: int,
        n_constr: int,
        eval_fn: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
        termination: Optional[Termination] = None,
        return_all: bool = False,
    ) -> NSGA2Result:
        xl_arr = np.asarray(xl, dtype=float).reshape(-1)
        xu_arr = np.asarray(xu, dtype=float).reshape(-1)
        if xl_arr.shape[0] != n_var or xu_arr.shape[0] != n_var:
            raise ValueError("xl/xu must have length n_var")

        problem = _CallbackProblem(
            n_var=n_var,
            xl=xl_arr,
            xu=xu_arr,
            n_obj=n_obj,
            n_constr=n_constr,
            eval_fn=eval_fn,
        )

        algorithm = NSGA2(
            pop_size=self.config.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(
                prob=self.config.crossover_prob, eta=self.config.crossover_eta
            ),
            mutation=PM(eta=self.config.mutation_eta),
            eliminate_duplicates=self.config.eliminate_duplicates,
        )

        if termination is None:
            termination = get_termination("n_gen", self.config.n_gen)

        res = minimize(
            problem,
            algorithm,
            termination,
            seed=self.config.seed,
            save_history=bool(return_all),
            verbose=False,
        )

        X = res.X
        F = res.F
        G = getattr(res, "G", None)
        feasible = None
        if n_constr > 0 and G is not None:
            feasible = np.all(G <= 0.0, axis=1)
        elif n_constr == 0:
            feasible = None

        best_idx = None
        if X is not None and F is not None and len(F) > 0:
            if feasible is not None:
                feas_idx = np.where(feasible)[0]
                if len(feas_idx) > 0:
                    # pick lexicographic best among feasible by objectives
                    f_feas = F[feas_idx]
                    order = np.lexsort(
                        [f_feas[:, i] for i in reversed(range(f_feas.shape[1]))]
                    )
                    best_idx = int(feas_idx[order[0]])
                else:
                    # pick minimal constraint violation first, then objectives
                    cv = np.maximum(G, 0.0).sum(axis=1)
                    best_idx = int(
                        np.lexsort(
                            [F[:, i] for i in reversed(range(F.shape[1]))] + [cv]
                        )[0]
                    )
            else:
                order = np.lexsort([F[:, i] for i in reversed(range(F.shape[1]))])
                best_idx = int(order[0])

        meta: Dict[str, Any] = {
            "algorithm": "NSGA2",
            "pop_size": int(self.config.pop_size),
            "n_gen": int(self.config.n_gen),
            "seed": int(self.config.seed),
            "crossover_prob": float(self.config.crossover_prob),
            "crossover_eta": float(self.config.crossover_eta),
            "mutation_eta": float(self.config.mutation_eta),
            "eliminate_duplicates": bool(self.config.eliminate_duplicates),
            "n_evals": int(getattr(res, "evals", -1))
            if hasattr(res, "evals")
            else None,
            "has_constraints": bool(n_constr > 0),
            "return_all": bool(return_all),
        }
        if return_all and hasattr(res, "history") and res.history is not None:
            meta["history_len"] = len(res.history)

        return NSGA2Result(
            X=np.asarray(X) if X is not None else np.empty((0, n_var), dtype=float),
            F=np.asarray(F) if F is not None else np.empty((0, n_obj), dtype=float),
            G=np.asarray(G) if G is not None else None,
            feasible=np.asarray(feasible) if feasible is not None else None,
            best_idx=best_idx,
            meta=meta,
        )
