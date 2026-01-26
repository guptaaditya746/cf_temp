# runner.py
# End-to-end Latent-Space Counterfactual Runner
#
# Wires Parts 1–6 into ONE callable entry point.
#
# Supports:
# - Scalar + CMA-ES (pycma)
# - Multi-objective + NSGA-II (pymoo)
#
# You provide:
# - encoder(x)->z, decoder(z)->x_hat, score_fn(x)->float
# - decoded constraint spec (Part 2)
# - NormalCore_z (recommended)
#
# Returns final standardized output:
# {
#   "x_cf": Tensor(L,F) or None,
#   "z_cf": Tensor(D,) or None,
#   "score": float or None,
#   "meta": {...}   # success/failure details
# }
#
# NOTE: This file assumes you have the Part modules available.
# Use package-relative imports in a real repo.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
import torch

# Part 2
from methods.latent.decoded_constraints import (
    DecodedConstraintEvaluator,
    DecodedConstraintSpec,
)

# Part 6
from methods.latent.failure_handling import assemble_final_result

# ---- Import your modules (adjust imports to your package layout) ----
# Part 1
from methods.latent.latent_problem import LatentCFProblem, define_latent_cf_problem

# Part 3
from methods.latent.objectives import (
    MultiObjectiveLatentCF,
    ScalarLatentCFObjective,
    ScalarObjectiveConfig,
)

# Part 4
from methods.latent.optimizers import (
    CMAESConfig,
    CMAESLatentOptimizer,
    NSGA2Config,
    NSGA2LatentOptimizer,
)

# Part 5
from methods.latent.selection_and_validation import (
    SelectionConfig,
    select_from_candidates,
    select_from_pareto,
)

Tensor = torch.Tensor


# -------------------------
# Runner configuration
# -------------------------


@dataclass
class RunnerConfig:
    """
    High-level runner configuration.

    mode:
      - "scalar_cmaes": scalar objective + CMA-ES
      - "moo_nsga2": multi-objective + NSGA-II
    """

    mode: Literal["scalar_cmaes", "moo_nsga2"] = "scalar_cmaes"

    # Validity margin beyond tau
    eps_validity: float = 1e-3

    # Search-space locality (latent L2 ball radius); strongly recommended
    latent_eps: Optional[float] = None

    # Optional latent bounds (D,2) and edit mask (D,)
    bounds: Optional[Tensor] = None
    editable_mask: Optional[Tensor] = None

    # Objective configs
    scalar_obj_cfg: ScalarObjectiveConfig = field(default_factory=ScalarObjectiveConfig)
    moo_normalcore_reduction: Literal["min", "mean"] = "min"

    # Optimizer configs
    cmaes_cfg: CMAESConfig = field(default_factory=CMAESConfig)  # ← FIX THIS LINE
    nsga2_cfg: NSGA2Config = field(default_factory=NSGA2Config)  # ← AND THIS ONE

    # Selection configs
    selection_cfg: SelectionConfig = field(default_factory=SelectionConfig)
    pareto_strategy: Literal["knee", "min_latent"] = "knee"

    # Failure classifier hint
    latent_jump_threshold: Optional[float] = None  # if set, flags regime teleportation

    # Randomness
    seed: Optional[int] = None


# -------------------------
# Public API
# -------------------------


def generate_latent_counterfactual(
    *,
    x: Tensor,  # (L,F)
    encoder: Callable[[Tensor], Tensor],
    decoder: Callable[[Tensor], Tensor],
    score_fn: Callable[[Tensor], float],
    tau: float,
    constraint_spec: DecodedConstraintSpec,
    normalcore_z: Optional[Tensor] = None,
    cfg: RunnerConfig = RunnerConfig(),
) -> Dict[str, Any]:
    """
    End-to-end latent CF generation.

    Returns the FINAL standardized output dict (success or failure).
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"x must have shape (L,F). Got {tuple(x.shape)}")

    # -------------------------
    # Part 1: Problem Definition
    # -------------------------
    problem: LatentCFProblem = define_latent_cf_problem(
        encoder=encoder,
        decoder=decoder,
        score_fn=score_fn,
        tau=float(tau),
        eps_validity=float(cfg.eps_validity),
        normalcore_z=normalcore_z,
        bounds=cfg.bounds,
        editable_mask=cfg.editable_mask,
        latent_eps=cfg.latent_eps,
        meta={"runner_mode": cfg.mode},
    ).build(x)

    x_orig = problem.x  # original window

    # -------------------------
    # Part 2: Constraints (decoded)
    # -------------------------
    constraint_eval = DecodedConstraintEvaluator(constraint_spec)

    # -------------------------
    # Part 3: Objective(s)
    # -------------------------
    scalar_objective = ScalarLatentCFObjective(config=cfg.scalar_obj_cfg)
    multi_objective = MultiObjectiveLatentCF(
        normalcore_reduction=cfg.moo_normalcore_reduction
    )

    # -------------------------
    # Part 4: Optimization
    # -------------------------
    candidates_history = []
    pareto_set = None
    optimizer_name = ""

    if cfg.mode == "scalar_cmaes":
        optimizer_name = "CMA-ES(pycma)"
        cma_opt = CMAESLatentOptimizer(config=cfg.cmaes_cfg)
        out = cma_opt.run(
            problem=problem,
            constraint_eval=constraint_eval,
            scalar_objective=scalar_objective,
            x_orig=x_orig,
        )
        candidates_history = out["history"]

        # -------------------------
        # Part 5: Selection & Validation
        # -------------------------
        cf = select_from_candidates(
            candidates=candidates_history,
            problem=problem,
            constraint_eval=constraint_eval,
            x_orig=x_orig,
            cfg=cfg.selection_cfg,
            rng=np.random.default_rng(cfg.seed),
        )

        # -------------------------
        # Part 6: Failure handling & final assembly
        # -------------------------
        return assemble_final_result(
            cf=cf,
            candidates=candidates_history,
            problem=problem,
            optimizer_name=optimizer_name,
            latent_jump_threshold=cfg.latent_jump_threshold,
        )

    elif cfg.mode == "moo_nsga2":
        optimizer_name = "NSGA-II(pymoo)"
        nsga_opt = NSGA2LatentOptimizer(config=cfg.nsga2_cfg)
        out = nsga_opt.run(
            problem=problem,
            constraint_eval=constraint_eval,
            multi_objective=multi_objective,
            x_orig=x_orig,
        )
        candidates_history = out["history"]
        pareto_set = out["pareto"]

        # -------------------------
        # Part 5: Selection & Validation (Pareto)
        # -------------------------
        cf = select_from_pareto(
            pareto=pareto_set,
            problem=problem,
            constraint_eval=constraint_eval,
            x_orig=x_orig,
            cfg=cfg.selection_cfg,
            strategy=cfg.pareto_strategy,
        )

        # -------------------------
        # Part 6: Failure handling & final assembly
        # -------------------------
        # Use pareto_set if present, else fallback to history for failure classification
        failure_pool = pareto_set if pareto_set else candidates_history
        return assemble_final_result(
            cf=cf,
            candidates=failure_pool,
            problem=problem,
            optimizer_name=optimizer_name,
            latent_jump_threshold=cfg.latent_jump_threshold,
        )

    else:
        raise ValueError("cfg.mode must be 'scalar_cmaes' or 'moo_nsga2'")


# # -------------------------
# # Wrapper Class for Common API
# # -------------------------


# class LatentSpaceCounterfactual:
#     """
#     Latent-space counterfactual generator wrapped to match the common CF API.

#     Usage:
#         cf_method = LatentSpaceCounterfactual(
#             model=model,
#             threshold=threshold,
#             normal_windows=normal_windows,
#             device=device,
#             encoder=encoder_fn,
#             decoder=decoder_fn,
#             constraint_spec=constraint_spec,
#             cfg=cfg,  # optional
#         )
#         result = cf_method.generate(x_anomaly)
#     """

#     def __init__(
#         self,
#         model,
#         threshold,
#         normal_windows,
#         device,
#         encoder,
#         decoder,
#         constraint_spec,
#         cfg=None,
#     ):
#         """
#         Args:
#             model: Anomaly detection model (for score_fn)
#             threshold: Validity threshold (tau)
#             normal_windows: Normal examples for reference (B, L, F)
#             device: torch device
#             encoder: Function x → z
#             decoder: Function z → x_hat
#             constraint_spec: DecodedConstraintSpec instance
#             cfg: RunnerConfig (optional, defaults to scalar CMA-ES)
#         """
#         self.model = model
#         self.threshold = float(threshold)
#         self.device = device
#         self.encoder = encoder
#         self.decoder = decoder
#         self.constraint_spec = constraint_spec

#         # Default config if not provided
#         if cfg is None:
#             cfg = RunnerConfig(
#                 mode="scalar_cmaes",
#                 eps_validity=0.05,
#                 latent_eps=1.5,
#                 seed=42,
#                 cmaes_cfg=CMAESConfig(
#                     sigma0=0.3, max_evals=500, stop_on_first_valid=False
#                 ),
#                 selection_cfg=SelectionConfig(
#                     require_validity=True,
#                     robustness_trials=6,
#                     robustness_sigma=0.03,
#                     robustness_valid_frac=0.67,
#                 ),
#             )
#         self.cfg = cfg

#         # Encode normal windows to latent space
#         self.normalcore_z = self._encode_normal_core(normal_windows)

#     def _encode_normal_core(self, normal_windows):
#         """Encode normal windows to latent space (B, D)"""
#         with torch.no_grad():
#             # Ensure shape is (B, L, F)
#             if normal_windows.ndim == 2:
#                 normal_windows = normal_windows.unsqueeze(0)

#             z_list = []
#             for i in range(normal_windows.shape[0]):
#                 z = self.encoder(normal_windows[i].to(self.device))
#                 z_list.append(z.cpu())

#             return torch.stack(z_list, dim=0)

#     def _score_fn(self, x):
#         """Compute anomaly score for window x"""
#         with torch.no_grad():
#             x_device = x.to(self.device).unsqueeze(0)  # (1, L, F)
#             score = self.model.decision_function(x_device.cpu().numpy())
#             return float(score[0])

#     def generate(self, x_anomaly):
#         """
#         Generate counterfactual for x_anomaly.

#         Args:
#             x_anomaly: Tensor of shape (L, F)

#         Returns:
#             Result dict with keys: x_cf, z_cf, score, meta
#             Returns None if generation fails
#         """
#         # Ensure input is on correct device
#         x = x_anomaly.to(self.device)

#         # Compute original score
#         score_orig = self._score_fn(x)

#         # Generate counterfactual
#         result = generate_latent_counterfactual(
#             x=x,
#             encoder=self.encoder,
#             decoder=self.decoder,
#             score_fn=self._score_fn,
#             tau=self.threshold,
#             constraint_spec=self.constraint_spec,
#             normalcore_z=self.normalcore_z,
#             cfg=self.cfg,
#         )

#         # Add original score to meta
#         result["meta"]["score_orig"] = score_orig

#         # Add evaluation count if available
#         if "n_evals" in result["meta"]:
#             result["meta"]["evals"] = result["meta"]["n_evals"]

#         # Return None if failed (to match other methods' API)
#         if result["meta"]["status"] != "success":
#             print(
#                 f"⚠️  CF generation failed: {result['meta'].get('failure_type', 'unknown')}"
#             )
#             return None

#         return result


# -------------------------
# Minimal example (smoke test)
# -------------------------
if __name__ == "__main__":
    L, F, D = 32, 3, 8
    torch.manual_seed(0)

    class DummyAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(L * F, D)
            )
            self.dec = torch.nn.Sequential(torch.nn.Linear(D, L * F))

        def encode(self, x):
            return self.enc(x)

        def decode(self, z):
            return self.dec(z).view(L, F)

    ae = DummyAE()

    def encoder(x: Tensor) -> Tensor:
        with torch.no_grad():
            # Add batch dim if needed
            if x.ndim == 2:  # (L, F)
                x = x.unsqueeze(0)  # (1, L, F)
                z = ae.encode(x)
                return z.squeeze(0)  # (D,)
            else:  # Already batched
                return ae.encode(x)

    def decoder(z: Tensor) -> Tensor:
        with torch.no_grad():
            # Add batch dim if needed
            if z.ndim == 1:  # (D,)
                z = z.unsqueeze(0)  # (1, D)
                x_hat = ae.decode(z)
                return x_hat.squeeze(0)  # (L, F)
            else:  # Already batched
                return ae.decode(z)

    def score_fn(x: Tensor) -> float:
        # toy anomaly score: L2 magnitude
        return float(torch.mean(x**2).sqrt().item())

    x = torch.randn(L, F)
    normalcore_x = torch.randn(64, L, F)
    with torch.no_grad():
        normalcore_z = torch.stack(
            [encoder(normalcore_x[i]) for i in range(normalcore_x.shape[0])], dim=0
        )

    # Constraints
    constraint_spec = DecodedConstraintSpec(
        value_min=torch.tensor([-5.0, -5.0, -5.0]),
        value_max=torch.tensor([+5.0, +5.0, +5.0]),
        max_rate_of_change=torch.tensor([1.5, 1.5, 1.5]),
        smoothness_weight=1.0,
        roc_weight=1.0,
    )

    # Runner config: scalar CMA-ES
    cfg = RunnerConfig(
        mode="scalar_cmaes",
        eps_validity=0.05,
        latent_eps=1.5,
        seed=0,
        cmaes_cfg=CMAESConfig(sigma0=0.3, max_evals=400, stop_on_first_valid=False),
        selection_cfg=SelectionConfig(
            require_validity=True,
            robustness_trials=6,
            robustness_sigma=0.03,
            robustness_valid_frac=0.67,
        ),
    )

    # Pick a threshold so it can succeed in toy setup
    tau = 1.0

    result = generate_latent_counterfactual(
        x=x,
        encoder=encoder,
        decoder=decoder,
        score_fn=score_fn,
        tau=tau,
        constraint_spec=constraint_spec,
        normalcore_z=normalcore_z,
        cfg=cfg,
    )
    # print(result["x_cf"])
    print("RESULT status:", result["meta"]["status"])
    print(
        "RESULT meta:",
        {
            k: result["meta"].get(k)
            for k in [
                "status",
                "failure_type",
                "score",
                "validity_target",
                "optimizer",
                "robust",
            ]
        },
    )
