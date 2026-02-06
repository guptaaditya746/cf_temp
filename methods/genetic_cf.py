# cf_final/methods/genetic_cf.py
from __future__ import annotations

# Enables postponed evaluation of type annotations.
# This allows forward references and reduces runtime overhead of typing.
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# NSGA-II optimizer and configuration
from methods.genetic.optimizer_nsga2 import NSGA2Config, NSGA2Optimizer

# Domain-specific constraint system for sensor signals
from methods.genetic.sensor_constraints import SensorConstraintManager

# ============================================================
# Genome definition (problem-specific, NOT optimizer-specific)
# ============================================================


@dataclass(frozen=True)
class SegmentGenome:
    """
    Genome representation for a segment-level counterfactual.

    Each genome is continuous in [0,1] and later decoded into
    discrete indices.

    Fields:
    - start: normalized start position of the segment
    - length: normalized segment length
    - motif_id: normalized index selecting a motif from normal_core
    """

    start: float  # normalized [0,1] → mapped to time index
    length: float  # normalized [0,1] → mapped to segment length
    motif_id: float  # normalized [0,1] → mapped to motif index


# ============================================================
# Counterfactual Problem Definition
# ============================================================


class SegmentCounterfactualProblem:
    """
    Domain-specific counterfactual optimization problem
    for sensor time-series data.

    This class acts as the *bridge* between:
    - the black-box anomaly model
    - the domain constraints
    - the evolutionary optimizer (NSGA-II)

    Responsibilities:
    - Define genome encoding / decoding
    - Construct counterfactual signals x_cf from genomes
    - Enforce hard and soft sensor constraints
    - Compute multi-objective fitness vectors
    - Interface with NSGA-II optimizer
    """

    def __init__(
        self,
        *,
        model,
        threshold: float,
        normal_core: torch.Tensor,  # (K, L, F): reference normal windows
        constraints: SensorConstraintManager,
        optimizer: NSGA2Optimizer,
        device: str = "cpu",
        eps_valid: float = 0.0,
    ):
        # Black-box anomaly detection / reconstruction model
        self.model = model

        # Decision threshold for anomaly validity
        self.threshold = float(threshold)

        # Numerical slack for validity (allows soft acceptance)
        self.eps_valid = float(eps_valid)

        # Torch device (CPU / GPU)
        self.device = device

        # Normal reference windows used as motif source
        self.normal_core = normal_core.to(device)

        # Dimensions:
        # K = number of normal windows
        # L = window length (time)
        # F = number of features / sensors
        self.K, self.L, self.F = self.normal_core.shape

        # Constraint manager (hard + soft constraints)
        self.constraints = constraints

        # Multi-objective evolutionary optimizer
        self.optimizer = optimizer

        # Precompute all possible motifs from normal_core
        self._build_motif_index()

    # --------------------------------------------------------
    # Motif utilities
    # --------------------------------------------------------

    def _build_motif_index(self):
        """
        Precompute a flattened index of all possible motifs.

        Each motif is represented as:
        (window_id k, start_index, max_possible_length)

        This avoids recomputing valid motif ranges during optimization.
        """
        self.motifs: List[Tuple[int, int, int]] = []

        for k in range(self.K):
            for start in range(self.L):
                max_len = self.L - start
                if max_len <= 1:
                    continue  # skip degenerate motifs
                self.motifs.append((k, start, max_len))

    def _select_motif(self, u: float) -> Tuple[int, int, int]:
        """
        Map a normalized scalar u ∈ [0,1] to a valid motif.

        This converts a continuous genome value into
        a discrete motif choice.
        """
        idx = int(u * len(self.motifs))
        idx = max(0, min(len(self.motifs) - 1, idx))
        return self.motifs[idx]

    # --------------------------------------------------------
    # Genome → Counterfactual signal
    # --------------------------------------------------------

    def decode(self, genome: SegmentGenome, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a genome into a concrete counterfactual signal.

        Steps:
        1. Convert normalized genome values into discrete indices
        2. Select motif from normal_core
        3. Replace a contiguous segment in x
        """
        # Start from original signal
        x_cf = x.clone()

        # Map normalized start position to discrete index
        start = int(genome.start * (self.L - 1))

        # Select motif metadata
        motif_k, motif_start, max_len = self._select_motif(genome.motif_id)

        # Map normalized length to discrete length (minimum length = 2)
        length = max(2, int(genome.length * max_len))

        # Ensure segment stays within bounds
        end = min(self.L, start + length)

        # Extract motif segment
        motif_seg = self.normal_core[
            motif_k,
            motif_start : motif_start + (end - start),
        ]

        # Replace segment
        x_cf[start:end] = motif_seg
        return x_cf

    # --------------------------------------------------------
    # Evaluation for NSGA-II
    # --------------------------------------------------------

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized evaluation function for NSGA-II.

        Inputs:
        - X: population matrix (n_individuals × genome_dim)

        Outputs:
        - F: objective matrix (n × n_obj)
        - G: constraint violation matrix (n × n_constr)
        """
        n = X.shape[0]

        # Objective matrix:
        # [L1 edit, soft dynamics, soft coupling, normal distance, score excess]
        F = np.zeros((n, 5), dtype=float)

        # Constraint matrix:
        # single feasibility constraint (score validity)
        G = np.zeros((n, 1), dtype=float)

        # local copies for speed
        x0 = self._x

        for i in range(n):
            # Decode genome
            g = SegmentGenome(*X[i])
            x_cf = self.decode(g, x0)
            x_cf = self.constraints.repair(x0, x_cf)
            # -------------------------
            # Hard constraints
            # -------------------------
            hard = self.constraints.hard_check(x0, x_cf)
            if not hard.ok:
                # Hard violation → infeasible solution
                G[i, 0] = 1.0
                F[i, :] = np.inf
                # print("HARD FAIL:", hard.reasons)
                # Optional: show how far out of bounds
                # below = (x_cf < self.constraints.val_lo.unsqueeze(0)).sum().item()
                # above = (x_cf > self.constraints.val_hi.unsqueeze(0)).sum().item()
                # print("  below count:", below, "above count:", above)
                if self._event_cb is not None and self._debug:
                    self._event_cb(
                        {
                            "type": "genetic_hard_fail",
                            "reasons": list(hard.reasons),
                        }
                    )
                continue

            # -------------------------
            # Validity objective
            # -------------------------
            score = float(self.model.score(x_cf))
            score_excess = score - (self.threshold - self.eps_valid)  # <= 0 is feasible
            G[i, 0] = float(score_excess)
            # -------------------------
            # Soft constraints
            # -------------------------
            soft = self.constraints.soft_metrics(x_cf)

            # -------------------------
            # Minimality objective
            # -------------------------
            delta = (x_cf - self._x).abs()
            edit_l1 = float(delta.sum().item())

            # -------------------------
            # Distance to normal manifold
            # -------------------------
            diff = self.normal_core - x_cf.unsqueeze(0)
            normal_dist = float((diff**2).mean(dim=(1, 2)).min().item())

            # -------------------------
            # Multi-objective vector (all minimized)
            # -------------------------
            # ---- objective vector (all minimized) ----
            # Note: score_excess as objective should be nonnegative in many setups,
            # but we keep raw (can be negative if very valid). If you prefer, use max(0,score_excess).
            F[i, :] = [
                edit_l1,
                float(soft.dyn_violation + soft.curvature_violation),
                float(soft.coupling_violation),
                normal_dist,
                float(max(0.0, score_excess)),  # objective version
            ]

        return F, G

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def generate(
        self,
        x: torch.Tensor,
        *,
        return_all: bool = True,
        debug: bool = True,
        event_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns dict with:
          - x_cf
          - score
          - meta: includes objectives, genome, and (if return_all) optimizer trace
        """
        x = x.to(self.device)
        self._x = x
        self._debug = bool(debug)
        self._event_cb = event_cb

        base_score = float(self.model.score(x))

        if base_score <= (self.threshold - self.eps_valid):
            return {
                "x_cf": x.clone(),
                "score": base_score,
                "meta": {
                    "method": "segment_nsga2_cf",
                    "already_valid": True,
                    "base_score": base_score,
                    "threshold": self.threshold,
                },
            }

        # Run NSGA-II
        result = self.optimizer.run(
            n_var=3,
            xl=[0.0, 0.0, 0.0],
            xu=[1.0, 1.0, 1.0],
            n_obj=5,
            n_constr=1,
            eval_fn=self._evaluate_batch,
            return_all=bool(return_all),
        )

        if result.best_idx is None or result.X is None or len(result.X) == 0:
            return None

        best_idx = int(result.best_idx)
        best_genome = SegmentGenome(*result.X[best_idx])

        x_cf = self.decode(best_genome, x)
        x_cf = self.constraints.repair(x, x_cf)
        score_cf = float(self.model.score(x_cf))

        meta: Dict[str, Any] = {
            "method": "segment_nsga2_cf",
            "base_score": base_score,
            "score_cf": score_cf,
            "threshold": self.threshold,
            "best_idx": best_idx,
            "objectives": result.F[best_idx].tolist()
            if result.F is not None and len(result.F) > 0
            else None,
            "genome": dataclasses.asdict(best_genome),
            "optimizer_meta": result.meta,
        }

        # Put trace in a stable place for Stage B
        if return_all and isinstance(result.meta, dict) and "trace" in result.meta:
            meta["trace"] = result.meta["trace"]

        return {
            "x_cf": x_cf,
            "score": score_cf,
            "meta": meta,
        }
