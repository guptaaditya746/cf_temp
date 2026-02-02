from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils.io import load_npz, load_yaml, save_npz, save_yaml
from ..utils.model_io import load_torchscript
from ..utils.score import score_bundle


@dataclass(frozen=True)
class CFOut:
    normal_core_path: Path
    cf_results_path: Path
    cf_metadata_path: Path


def build_normal_core(train_x: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    """Sample normal core from training data."""
    N = train_x.shape[0]
    if N <= max_items:
        return train_x.astype(np.float32, copy=False)

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(N, size=int(max_items), replace=False)
    return train_x[idx].astype(np.float32, copy=False)


def score_windows(
    model, device: str, x_np: np.ndarray, agg: str, batch_size: int
) -> np.ndarray:
    """Compute reconstruction scores for windows."""
    dl = DataLoader(
        TensorDataset(torch.from_numpy(x_np)),
        batch_size=batch_size,
        shuffle=False,
    )
    scores = []

    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            xh = model(xb)
            b = score_bundle(xb, xh, aggregation=agg)
            scores.append(b["window"].detach().cpu().numpy())

    return np.concatenate(scores, axis=0).astype(np.float32)


def run_counterfactuals(run_dir: Path, cfg: Dict[str, Any]) -> CFOut:
    """Generate counterfactuals using configured method."""
    art = run_dir / "artifacts"

    # Load data
    targets = load_npz(art / "targets.npz")
    x_t = targets["x"].astype(np.float32)  # (M, L, F)
    idx_t = targets["idx"].astype(np.int64)  # (M,)
    score_t = targets["score"].astype(np.float32)  # (M,)

    train_x = load_npz(art / "train.npz")["x"].astype(np.float32)

    thr = load_yaml(art / "threshold.yaml")
    tau = float(thr["tau"])

    # Load model
    mcfg = cfg["model"]
    device = (
        "cuda"
        if str(mcfg.get("device", "cpu")) == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    model = load_torchscript(art / "model_ts.pt", device=device)

    agg = str(cfg.get("score", {}).get("aggregation", "mean"))
    bs = int(mcfg.get("batch_size", 256))

    # Build normal core
    cf_cfg = cfg["counterfactuals"]
    max_items = int(cf_cfg.get("normal_core", {}).get("max_items", 2000))
    seed = int(cf_cfg.get("seed", cfg.get("seed", {}).get("seed", 42)))

    normal_core = build_normal_core(train_x, max_items=max_items, seed=seed)
    normal_core_path = art / "normal_core.npz"
    save_npz(normal_core_path, x=normal_core)

    # Get CF method
    method = str(cf_cfg.get("method", "nearest_normal_window"))

    print(f"Generating counterfactuals using method: {method}")
    print(f"Targets: {x_t.shape[0]}, Threshold: {tau:.6f}")

    # ========================================================================
    # METHOD REGISTRY - All 7 CF Methods
    # ========================================================================

    if method == "segment_substitution":
        from methods.segment_substitution import SegmentSubstitutionCounterfactual

        cf_method = SegmentSubstitutionCounterfactual(
            model=model,
            threshold=tau,
            normal_windows=torch.from_numpy(normal_core).to(device),
            segment_length=int(cf_cfg.get("segment_length", 8)),
            device=device,
        )

        results = []
        for i in range(x_t.shape[0]):
            x_i = torch.from_numpy(x_t[i]).to(device)
            res = cf_method.generate(x_i)
            if res is not None:
                res["x_cf"] = res.get("xcf", res.get("x_cf")).cpu().numpy()
            results.append(res)

    elif method == "nearest_prototype":
        from methods.nearest_prototype import NearestPrototypeCounterfactual

        cf_method = NearestPrototypeCounterfactual(
            model=model,
            threshold=tau,
            normal_windows=torch.from_numpy(normal_core).to(device),
            device=device,
        )

        results = []
        for i in range(x_t.shape[0]):
            x_i = torch.from_numpy(x_t[i]).to(device)
            res = cf_method.generate(x_i)
            if res is not None:
                res["x_cf"] = res.get("xcf", res.get("x_cf")).cpu().numpy()
            results.append(res)

    elif method == "motif":
        from methods.motif import MotifGuidedSegmentRepairCF

        cf_method = MotifGuidedSegmentRepairCF(
            model=model,
            threshold=tau,
            device=device,
            normal_core=torch.from_numpy(normal_core).to(device),
            max_segments_per_len=int(cf_cfg.get("max_segments_per_len", 12)),
            top_motifs_per_segment=int(cf_cfg.get("top_motifs_per_segment", 8)),
            lengths=cf_cfg.get("lengths", [4, 6, 8, 10, 12, 16, 20, 24]),
            edge_blend=int(cf_cfg.get("edge_blend", 2)),
            use_error_guidance=cf_cfg.get("use_error_guidance", True),
        )

        results = []
        for i in range(x_t.shape[0]):
            x_i = torch.from_numpy(x_t[i]).to(device)
            res = cf_method.generate(x_i)
            if res is not None:
                res["x_cf"] = res.get("xcf", res.get("x_cf")).cpu().numpy()
            results.append(res)

    elif method == "genetic":
        from methods.genetic_cf import SegmentCounterfactualProblem
        from methods.genetic.optimizer_nsga2 import NSGA2Config, NSGA2Optimizer
        from methods.genetic.sensor_constraints import SensorConstraintManager

        # Setup constraints
        constraints = SensorConstraintManager(
            normal_core=torch.from_numpy(normal_core).to(device),
            value_quantiles=(0.01, 0.99),
            max_delta_quantile=0.99,
            device=str(device),
        )

        # Setup optimizer
        nsga2_config = NSGA2Config(
            pop_size=int(cf_cfg.get("pop_size", 100)),
            n_gen=int(cf_cfg.get("n_gen", 200)),
            seed=seed,
            crossover_prob=float(cf_cfg.get("crossover_prob", 0.4)),
        )
        optimizer = NSGA2Optimizer(nsga2_config)

        # Setup CF problem
        cf_method = SegmentCounterfactualProblem(
            model=model,
            threshold=tau,
            normal_core=torch.from_numpy(normal_core).to(device),
            constraints=constraints,
            optimizer=optimizer,
            device=str(device),
            eps_valid=float(cf_cfg.get("eps_valid", 0.5)),
        )

        results = []
        for i in range(x_t.shape[0]):
            x_i = torch.from_numpy(x_t[i]).to(device)
            res = cf_method.generate(x_i)
            if res is not None:
                res["x_cf"] = res.get("xcf", res.get("x_cf")).cpu().numpy()
            results.append(res)

    elif method == "latent":
        from methods.latent_final import LatentSpaceCounterfactual
        from methods.latent.decoded_constraints import DecodedConstraintSpec
        from methods.latent.latent_cf import RunnerConfig
        from methods.latent.optimizers import CMAESConfig

        # Compute data-driven constraints
        mins = torch.tensor(train_x.min(axis=(0, 1)), dtype=torch.float32)
        maxs = torch.tensor(train_x.max(axis=(0, 1)), dtype=torch.float32)
        roc = torch.tensor(
            np.percentile(np.diff(train_x, axis=1), 99, axis=(0, 1)),
            dtype=torch.float32,
        )

        constraint = DecodedConstraintSpec(
            value_min=mins - 0.5,
            value_max=maxs + 0.5,
            max_rate_of_change=roc,
            smoothness_weight=1.0,
            roc_weight=1.0,
        )

        # Create encoder/decoder wrappers (simplified - adapt to your model)
        def encoder_wrapper(x):
            with torch.no_grad():
                if hasattr(model, "encoder"):
                    _, (h_n, _) = model.encoder(x.unsqueeze(0))
                    return h_n[-1]
                else:
                    # Fallback: use middle layer
                    return model(x.unsqueeze(0)).mean(dim=1)

        def decoder_wrapper(z):
            # Simplified decoder - adapt to your model architecture
            return model(
                z.unsqueeze(0).unsqueeze(1).repeat(1, x_t.shape[1], 1)
            ).squeeze(0)

        cf_method = LatentSpaceCounterfactual(
            model=model,
            threshold=tau,
            normal_windows=torch.from_numpy(normal_core[:200]).to(device),
            device=device,
            encoder=encoder_wrapper,
            decoder=decoder_wrapper,
            constraint_spec=constraint,
            cfg=RunnerConfig(
                mode="scalar_cmaes",
                latent_eps=float(cf_cfg.get("latent_eps", 0.5)),
                cmaes_cfg=CMAESConfig(max_evals=int(cf_cfg.get("max_evals", 500))),
            ),
        )

        results = []
        for i in range(x_t.shape[0]):
            x_i = torch.from_numpy(x_t[i]).to(device)
            res = cf_method.generate(x_i)
            if res is not None:
                res["x_cf"] = res.get("xcf", res.get("x_cf")).cpu().numpy()
            results.append(res)

    elif method == "generative":
        from methods.generative_final import GenerativeInfillingCounterfactual
        from methods.generative.mask_strategy import MaskStrategy, MaskStrategyConfig
        from methods.generative.infilling_engine import InfillingEngine, InfillingConfig
        from methods.generative.candidate_selection import (
            CandidateSelector,
            SelectionConfig,
        )
        from methods.generative.constraint_evaluation import (
            ConstraintEvaluator,
            ConstraintConfig,
        )

        # Setup components
        mask_strategy = MaskStrategy(
            MaskStrategyConfig(
                mode=cf_cfg.get("mask_mode", "error_guided"),
                initial_masks=int(cf_cfg.get("initial_masks", 5)),
            )
        )

        infilling_engine = InfillingEngine(
            InfillingConfig(
                strategy=cf_cfg.get("infill_strategy", "interpolation"),
                max_candidates=int(cf_cfg.get("max_candidates", 20)),
            )
        )

        constraint_eval = ConstraintEvaluator(
            normal_core=torch.from_numpy(normal_core).to(device),
            cfg=ConstraintConfig(
                sensor_min=float(np.min(train_x)),
                sensor_max=float(np.max(train_x)),
            ),
        )

        candidate_selector = CandidateSelector(
            SelectionConfig(
                max_segments=int(cf_cfg.get("max_segments", 3)),
            )
        )

        cf_method = GenerativeInfillingCounterfactual(
            model=model,
            threshold=tau,
            mask_strategy=mask_strategy,
            infilling_engine=infilling_engine,
            constraint_evaluator=constraint_eval,
            candidate_selector=candidate_selector,
            device=device,
        )

        results = []
        for i in range(x_t.shape[0]):
            x_i = torch.from_numpy(x_t[i]).to(device)
            res = cf_method.generate(x_i)
            if res is not None:
                res["x_cf"] = res.get("xcf", res.get("x_cf")).cpu().numpy()
            results.append(res)

    elif method == "comte":
        from methods.comte_final import CoMTECounterfactual

        cf_method = CoMTECounterfactual(
            model=model,
            threshold=tau,
            normal_core=torch.from_numpy(normal_core).to(device),
            device=device,
            max_segments_per_len=int(cf_cfg.get("max_segments_per_len", 12)),
            top_donors_per_segment=int(cf_cfg.get("top_donors_per_segment", 8)),
            lengths=cf_cfg.get("lengths", [8, 12, 16, 24]),
        )

        results = []
        for i in range(x_t.shape[0]):
            x_i = torch.from_numpy(x_t[i]).to(device)
            res = cf_method.generate(x_i)
            if res is not None:
                res["x_cf"] = res.get("xcf", res.get("x_cf")).cpu().numpy()
            results.append(res)

    else:
        raise ValueError(f"Unknown counterfactual method: {method}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    # Collect successful CFs
    cf_windows = []
    cf_scores = []
    cf_metadata = []

    for i, res in enumerate(results):
        if res is not None:
            cf_windows.append(res["x_cf"])
            cf_scores.append(res["score"])
            cf_metadata.append(
                {
                    "target_idx": int(idx_t[i]),
                    "original_score": float(score_t[i]),
                    "cf_score": float(res["score"]),
                    "valid": bool(res["score"] < tau),
                    "meta": res.get("meta", {}),
                }
            )
        else:
            cf_windows.append(np.zeros_like(x_t[i]))
            cf_scores.append(np.inf)
            cf_metadata.append(
                {
                    "target_idx": int(idx_t[i]),
                    "original_score": float(score_t[i]),
                    "cf_score": np.inf,
                    "valid": False,
                    "meta": {"failure": "no_cf_found"},
                }
            )

    cf_results_path = art / "cf_results.npz"
    save_npz(
        cf_results_path,
        x_cf=np.stack(cf_windows).astype(np.float32),
        score=np.array(cf_scores).astype(np.float32),
    )

    cf_metadata_path = art / "cf_metadata.yaml"
    save_yaml(
        cf_metadata_path,
        {
            "method": method,
            "threshold": tau,
            "n_targets": len(results),
            "n_successful": sum(1 for r in results if r is not None),
            "success_rate": sum(1 for r in results if r is not None) / len(results),
            "results": cf_metadata,
        },
    )

    print(
        f"✓ CF generation complete: {sum(1 for r in results if r is not None)}/{len(results)} successful"
    )
            "success_rate": sum(1 for r in results if r is not None) / len(results),
            "results": cf_metadata,
        },
    )

    print(f"✓ CF generation complete: {sum(1 for r in results if r is not None)}/{len(results)} successful")

    return CFOut(
        normal_core_path=normal_core_path,
        cf_results_path=cf_results_path,
        cf_metadata_path=cf_metadata_path,
    )
