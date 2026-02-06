# cf_final/methods/genetic/optimizer_nsga2.py
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
    pop_size: int = 200
    n_gen: int = 80
    seed: int = 7
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0

    verbose: bool = True
    save_history: bool = True
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
            elementwise_evaluation=True,
        )
        self._eval_fn = eval_fn

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = self._eval_fn(X)
        out["F"] = F
        if self.n_constr > 0:
            if G is None:
                raise ValueError(
                    "Problem expects constraints (n_constr > 0) but eval_fn returned None for G."
                )
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
        return_all: bool = True,
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
            verbose=True,
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

        # -------------------------
        # TRACE EXTRACTION (for plots)
        # -------------------------
        if return_all:
            trace: Dict[str, Any] = {}

            # Final population (res.pop)
            try:
                pop = getattr(res, "pop", None)
                if pop is not None:
                    X_final = pop.get("X")
                    F_final = pop.get("F")
                    G_final = pop.get("G") if n_constr > 0 else None
                    CV_final = pop.get("CV") if n_constr > 0 else None

                    trace["X_final"] = (
                        np.asarray(X_final) if X_final is not None else None
                    )
                    trace["F_final"] = (
                        np.asarray(F_final) if F_final is not None else None
                    )
                    trace["G_final"] = (
                        np.asarray(G_final) if G_final is not None else None
                    )
                    trace["CV_final"] = (
                        np.asarray(CV_final) if CV_final is not None else None
                    )
            except Exception as e:
                trace["final_pop_error"] = repr(e)

            # ND set (res.X/res.F)
            trace["X_nd"] = np.asarray(res.X) if res.X is not None else None
            trace["F_nd"] = np.asarray(res.F) if res.F is not None else None
            trace["G_nd"] = (
                np.asarray(getattr(res, "G", None))
                if getattr(res, "G", None) is not None
                else None
            )
            trace["CV_nd"] = (
                np.asarray(getattr(res, "CV", None))
                if getattr(res, "CV", None) is not None
                else None
            )

            # Per-generation traces from history
            hist = getattr(res, "history", None)
            if hist is not None:
                best_F = []
                cv_mean = []
                cv_min = []
                pop_sizes = []

                for h in hist:
                    pop = getattr(h, "pop", None)
                    if pop is None:
                        best_F.append(np.full((n_obj,), np.nan))
                        cv_mean.append(np.nan)
                        cv_min.append(np.nan)
                        pop_sizes.append(0)
                        continue

                    Fh = pop.get("F")
                    CVh = pop.get("CV") if n_constr > 0 else None

                    if Fh is not None and len(Fh) > 0:
                        Fh = np.asarray(Fh)
                        best_F.append(np.min(Fh, axis=0))
                        pop_sizes.append(int(Fh.shape[0]))
                    else:
                        best_F.append(np.full((n_obj,), np.nan))
                        pop_sizes.append(0)

                    if CVh is not None and len(CVh) > 0:
                        CVh = np.asarray(CVh).reshape(-1)
                        cv_mean.append(float(np.mean(CVh)))
                        cv_min.append(float(np.min(CVh)))
                    else:
                        cv_mean.append(np.nan)
                        cv_min.append(np.nan)

                trace["best_F_per_gen"] = np.asarray(best_F, dtype=float)
                trace["cv_per_gen_mean"] = np.asarray(cv_mean, dtype=float)
                trace["cv_per_gen_min"] = np.asarray(cv_min, dtype=float)
                trace["pop_size_per_gen"] = np.asarray(pop_sizes, dtype=int)

            # Attach to meta so Stage B can plot it
            meta["trace"] = trace

        return NSGA2Result(
            X=np.asarray(X) if X is not None else np.empty((0, n_var), dtype=float),
            F=np.asarray(F) if F is not None else np.empty((0, n_obj), dtype=float),
            G=np.asarray(G) if G is not None else None,
            feasible=np.asarray(feasible) if feasible is not None else None,
            best_idx=best_idx,
            meta=meta,
        )
