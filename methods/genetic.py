from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from optimizer_nsga2 import NSGA2Config, NSGA2Optimizer
from sensor_constraints import SensorConstraintManager

# ============================================================
# Genome definition (problem-specific, NOT optimizer-specific)
# ============================================================


@dataclass(frozen=True)
class SegmentGenome:
    start: float  # normalized [0,1]
    length: float  # normalized [0,1]
    motif_id: float  # normalized [0,1]


# ============================================================
# Counterfactual Problem Definition
# ============================================================


class SegmentCounterfactualProblem:
    """
    Domain-specific counterfactual problem for sensor time-series.

    Responsibilities:
    - Define genome encoding / decoding
    - Construct x_cf from genome
    - Call sensor constraints
    - Call black-box reconstruction score
    - Return multi-objective vector + constraints for NSGA-II
    """

    def __init__(
        self,
        *,
        model,
        threshold: float,
        normal_core: torch.Tensor,  # (K,L,F)
        constraints: SensorConstraintManager,
        optimizer: NSGA2Optimizer,
        device: str = "cpu",
        eps_valid: float = 0.0,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.eps_valid = float(eps_valid)
        self.device = device

        self.normal_core = normal_core.to(device)
        self.K, self.L, self.F = self.normal_core.shape

        self.constraints = constraints
        self.optimizer = optimizer

        self._build_motif_index()

    # --------------------------------------------------------
    # Motif utilities
    # --------------------------------------------------------
    def _build_motif_index(self):
        self.motifs: List[Tuple[int, int, int]] = []
        for k in range(self.K):
            for start in range(self.L):
                max_len = self.L - start
                if max_len <= 1:
                    continue
                self.motifs.append((k, start, max_len))

    def _select_motif(self, u: float) -> Tuple[int, int, int]:
        idx = int(u * len(self.motifs))
        idx = max(0, min(len(self.motifs) - 1, idx))
        return self.motifs[idx]

    # --------------------------------------------------------
    # Genome → x_cf
    # --------------------------------------------------------
    def decode(self, genome: SegmentGenome, x: torch.Tensor) -> torch.Tensor:
        x_cf = x.clone()

        start = int(genome.start * (self.L - 1))
        motif_k, motif_start, max_len = self._select_motif(genome.motif_id)
        length = max(2, int(genome.length * max_len))

        end = min(self.L, start + length)
        motif_seg = self.normal_core[motif_k, motif_start : motif_start + (end - start)]

        x_cf[start:end] = motif_seg
        return x_cf

    # --------------------------------------------------------
    # Evaluation for NSGA-II
    # --------------------------------------------------------
    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        F = np.zeros((n, 5), dtype=float)  # objectives
        G = np.zeros((n, 1), dtype=float)  # feasibility constraint

        for i in range(n):
            g = SegmentGenome(*X[i])
            x_cf = self.decode(g, self._x)

            # hard constraints
            hard = self.constraints.hard_check(self._x, x_cf)
            if not hard.ok:
                G[i, 0] = 1.0
                F[i, :] = np.inf
                continue

            # score validity
            score = float(self.model.score(x_cf))
            score_excess = max(0.0, score - (self.threshold - self.eps_valid))

            # soft constraints
            soft = self.constraints.soft_metrics(x_cf)

            # minimality
            delta = (x_cf - self._x).abs()
            edit_l1 = float(delta.sum().item())

            # distance to NormalCore (nearest window)
            diff = self.normal_core - x_cf.unsqueeze(0)
            normal_dist = float((diff**2).mean(dim=(1, 2)).min().item())

            # objectives (all minimized)
            F[i, :] = [
                edit_l1,
                soft.dyn_violation + soft.curvature_violation,
                soft.coupling_violation,
                normal_dist,
                score_excess,
            ]

            G[i, 0] = 0.0 if score_excess == 0.0 else score_excess

        return F, G

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def generate(self, x: torch.Tensor) -> Optional[Dict[str, Any]]:
        x = x.to(self.device)
        self._x = x

        base_score = float(self.model.score(x))
        if base_score <= (self.threshold - self.eps_valid):
            return {
                "x_cf": x.clone(),
                "score": base_score,
                "meta": {
                    "already_valid": True,
                    "base_score": base_score,
                },
            }

        result = self.optimizer.run(
            n_var=3,
            xl=[0.0, 0.0, 0.0],
            xu=[1.0, 1.0, 1.0],
            n_obj=5,
            n_constr=1,
            eval_fn=self._evaluate_batch,
        )

        if result.best_idx is None:
            return None

        best_genome = SegmentGenome(*result.X[result.best_idx])
        x_cf = self.decode(best_genome, x)
        score = float(self.model.score(x_cf))

        return {
            "x_cf": x_cf,
            "score": score,
            "meta": {
                "method": "segment_nsga2_cf",
                "base_score": base_score,
                "threshold": self.threshold,
                "objectives": result.F[result.best_idx].tolist(),
                "genome": dataclasses.asdict(best_genome),
            },
        }
