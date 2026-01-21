# cf_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    if x.dim() == 2:
        return x
    if x.dim() == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError(f"Expected (L,F) or (1,L,F), got {tuple(x.shape)}")


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x.unsqueeze(0)
    raise ValueError(f"Expected (K,L,F) or (L,F), got {tuple(x.shape)}")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return float(x.detach().reshape(-1)[0].cpu().item())
    try:
        return float(x)
    except Exception:
        return None


def _rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x - y) ** 2) + 1e-12)


def _mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def _max_abs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(x - y))


def _diff1(x: torch.Tensor) -> torch.Tensor:
    return x[1:] - x[:-1]


def _diff2(x: torch.Tensor) -> torch.Tensor:
    d1 = _diff1(x)
    return d1[1:] - d1[:-1]


def _segment_stats_from_mask(mask: torch.Tensor) -> Dict[str, Any]:
    # mask: (L,) bool
    L = int(mask.numel())
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return {
            "changed_any": False,
            "n_changed": 0,
            "frac_changed": 0.0,
            "n_segments": 0,
            "max_segment_len": 0,
            "min_segment_len": 0,
            "mean_segment_len": 0.0,
            "segment_lengths": [],
            "first_change": None,
            "last_change": None,
            "contiguous": True,
        }

    idx_sorted = torch.sort(idx).values
    diffs = idx_sorted[1:] - idx_sorted[:-1]
    breaks = torch.nonzero(diffs > 1, as_tuple=False).reshape(-1)

    seg_starts = torch.cat([idx_sorted[:1], idx_sorted[breaks + 1]])
    seg_ends = torch.cat([idx_sorted[breaks], idx_sorted[-1:]])  # inclusive
    seg_lens = (seg_ends - seg_starts + 1).to(torch.long)

    n_segs = int(seg_lens.numel())
    seg_lens_list = [int(v.item()) for v in seg_lens]
    max_len = int(seg_lens.max().item())
    min_len = int(seg_lens.min().item())
    mean_len = float(seg_lens.float().mean().item())

    return {
        "changed_any": True,
        "n_changed": int(idx_sorted.numel()),
        "frac_changed": float(idx_sorted.numel() / max(1, L)),
        "n_segments": n_segs,
        "max_segment_len": max_len,
        "min_segment_len": min_len,
        "mean_segment_len": mean_len,
        "segment_lengths": seg_lens_list,
        "first_change": int(idx_sorted[0].item()),
        "last_change": int(idx_sorted[-1].item()),
        "contiguous": (n_segs == 1),
    }


def _feature_bounds_from_normal_core(
    normal_core: torch.Tensor,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nc = _to_3d(normal_core)
    flat = nc.reshape(-1, nc.shape[-1])  # (K*L, F)
    lo = torch.quantile(flat, q_lo, dim=0)
    hi = torch.quantile(flat, q_hi, dim=0)
    eps = 1e-6
    hi = torch.maximum(hi, lo + eps)
    return lo, hi


def _nn_plausibility_dist(
    x_cf: torch.Tensor,
    normal_core: torch.Tensor,
    metric: str = "rmse",
    chunk: int = 256,
) -> float:
    """
    Returns distance from x_cf to nearest window in NormalCore (K,L,F)
    using RMSE or MAE. Chunked to reduce memory.
    """
    x2 = _to_2d(x_cf)
    nc = _to_3d(normal_core)
    K = nc.shape[0]

    best = None
    for i in range(0, K, chunk):
        batch = nc[i : i + chunk]  # (B,L,F)
        # (B,)
        if metric == "mae":
            d = torch.mean(torch.abs(batch - x2.unsqueeze(0)), dim=(1, 2))
        else:
            d = torch.sqrt(torch.mean((batch - x2.unsqueeze(0)) ** 2, dim=(1, 2)) + 1e-12)
        m = d.min()
        best = m if best is None else torch.minimum(best, m)

    return float(best.detach().cpu().item()) if best is not None else float("inf")


def _zscore_plausibility(
    x_cf: torch.Tensor,
    normal_core: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Per-feature z-score using NormalCore flattened across time and windows.
    Returns mean/max absolute z-score over ALL points.
    """
    x2 = _to_2d(x_cf)
    nc = _to_3d(normal_core)
    flat = nc.reshape(-1, nc.shape[-1])  # (K*L, F)
    mu = flat.mean(dim=0)
    sd = flat.std(dim=0).clamp_min(eps)
    z = (x2 - mu) / sd  # (L,F)
    absz = torch.abs(z)
    return {
        "z_abs_mean": float(absz.mean().detach().cpu().item()),
        "z_abs_max": float(absz.max().detach().cpu().item()),
    }


@dataclass(frozen=True)
class MetricsConfig:
    eps_change: float = 1e-6
    bounds_q_lo: float = 0.01
    bounds_q_hi: float = 0.99
    nn_metric: str = "rmse"  # "rmse" or "mae"
    nn_chunk: int = 256


class CounterfactualMetrics:
    def __init__(self, config: MetricsConfig = MetricsConfig()):
        self.cfg = config

    def compute(
        self,
        x: torch.Tensor,
        cf_result: Optional[Dict[str, Any]],
        threshold: float,
        normal_core: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        x: (L,F) or (1,L,F)
        cf_result: {"x_cf": Tensor(L,F), "score": float, "meta": dict} or None
        threshold: scalar
        normal_core: (K,L,F) optional, used for plausibility metrics
        """
        x0 = _to_2d(x).detach()
        thr = float(threshold)

        out: Dict[str, Any] = {
            "found": cf_result is not None,
            "threshold": thr,
        }

        if cf_result is None:
            out.update(
                {
                    "valid": False,
                    "score_cf": None,
                    "delta_score_to_thr": None,
                    "dist_rmse": None,
                    "dist_mae": None,
                    "dist_max_abs": None,
                    "n_changed": None,
                    "frac_changed": None,
                    "n_segments": None,
                    "max_segment_len": None,
                    "contiguous": None,
                    "smooth_l2_d1": None,
                    "smooth_l2_d2": None,
                    "bounds_violations": None,
                    "bounds_violation_frac": None,
                    "nn_dist_to_normal_core": None,
                    "z_abs_mean": None,
                    "z_abs_max": None,
                    "meta": None,
                }
            )
            return out

        x_cf = _to_2d(cf_result.get("x_cf")).detach()
        score_cf = _safe_float(cf_result.get("score"))
        meta = cf_result.get("meta", {})
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            meta = {"raw_meta": meta}

        out["meta"] = meta
        out["score_cf"] = score_cf
        if score_cf is not None:
            out["valid"] = bool(score_cf <= thr)
            out["delta_score_to_thr"] = float(score_cf - thr)
        else:
            out["valid"] = False
            out["delta_score_to_thr"] = None

        # Proximity
        out["dist_rmse"] = float(_rmse(x0, x_cf).cpu().item())
        out["dist_mae"] = float(_mae(x0, x_cf).cpu().item())
        out["dist_max_abs"] = float(_max_abs(x0, x_cf).cpu().item())

        # Change structure (temporal contiguity)
        delta = torch.abs(x_cf - x0)  # (L,F)
        changed_t = (delta.max(dim=1).values > self.cfg.eps_change)  # (L,)
        seg_stats = _segment_stats_from_mask(changed_t)
        out.update(seg_stats)

        # Smoothness of edits (penalize jagged edits over time)
        # computed on the delta (x_cf - x) so it measures edit smoothness, not signal smoothness
        d = (x_cf - x0)  # (L,F)
        if d.shape[0] >= 2:
            d1 = _diff1(d)
            out["smooth_l2_d1"] = float(torch.sqrt(torch.mean(d1**2) + 1e-12).detach().cpu().item())
        else:
            out["smooth_l2_d1"] = 0.0
        if d.shape[0] >= 3:
            d2 = _diff2(d)
            out["smooth_l2_d2"] = float(torch.sqrt(torch.mean(d2**2) + 1e-12).detach().cpu().item())
        else:
            out["smooth_l2_d2"] = 0.0

        # Budget / bookkeeping if present
        out["evals"] = _safe_float(meta.get("evals"))
        out["method"] = meta.get("method", meta.get("strategy", None))

        # Plausibility (NormalCore)
        if normal_core is None:
            out["bounds_violations"] = None
            out["bounds_violation_frac"] = None
            out["nn_dist_to_normal_core"] = None
            out["z_abs_mean"] = None
            out["z_abs_max"] = None
            return out

        nc = _to_3d(normal_core).detach()

        # Bounds violations using quantile bounds from NormalCore
        lo, hi = _feature_bounds_from_normal_core(
            nc,
            q_lo=self.cfg.bounds_q_lo,
            q_hi=self.cfg.bounds_q_hi,
        )
        viol = (x_cf < lo) | (x_cf > hi)  # broadcasts (L,F) vs (F,)
        n_viol = int(viol.sum().item())
        out["bounds_violations"] = n_viol
        out["bounds_violation_frac"] = float(n_viol / max(1, int(x_cf.numel())))

        # Nearest-neighbor distance to NormalCore
        out["nn_dist_to_normal_core"] = _nn_plausibility_dist(
            x_cf=x_cf,
            normal_core=nc,
            metric=self.cfg.nn_metric,
            chunk=self.cfg.nn_chunk,
        )

        # Z-score plausibility
        out.update(_zscore_plausibility(x_cf=x_cf, normal_core=nc))

        return out

    def compute_batch(
        self,
        xs: torch.Tensor,
        cf_results: List[Optional[Dict[str, Any]]],
        threshold: float,
        normal_core: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        xs: (B,L,F) or (L,F) with B inferred as 1
        cf_results: list length B
        """
        xs3 = _to_3d(xs) if xs.dim() != 2 else xs.unsqueeze(0)
        B = xs3.shape[0]
        if len(cf_results) != B:
            raise ValueError(f"cf_results length {len(cf_results)} != batch size {B}")

        return [
            self.compute(xs3[i], cf_results[i], threshold=threshold, normal_core=normal_core)
            for i in range(B)
        ]


def summarize_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates a list of per-sample metric dicts into summary stats.
    Deterministic; ignores None entries.
    """
    if not metrics:
        return {}

    def collect(key: str) -> List[float]:
        vals: List[float] = []
        for m in metrics:
            v = m.get(key, None)
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            else:
                fv = _safe_float(v)
                if fv is not None:
                    vals.append(float(fv))
        return vals

    def stats(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"mean": None, "median": None, "min": None, "max": None}
        t = torch.tensor(vals, dtype=torch.float64)
        return {
            "mean": float(t.mean().item()),
            "median": float(t.median().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    keys = [
        "found",
        "valid",
        "score_cf",
        "delta_score_to_thr",
        "dist_rmse",
        "dist_mae",
        "dist_max_abs",
        "n_changed",
        "frac_changed",
        "n_segments",
        "max_segment_len",
        "smooth_l2_d1",
        "smooth_l2_d2",
        "bounds_violations",
        "bounds_violation_frac",
        "nn_dist_to_normal_core",
        "z_abs_mean",
        "z_abs_max",
        "evals",
    ]

    out: Dict[str, Any] = {
        "n": len(metrics),
        "found_rate": stats(collect("found"))["mean"],
        "valid_rate": stats(collect("valid"))["mean"],
    }

    for k in keys:
        if k in ("found", "valid"):
            continue
        out[k] = stats(collect(k))

    return out


def stability_metrics(
    per_seed_metrics: List[Dict[str, Any]],
    keys: Tuple[str, ...] = ("score_cf", "dist_rmse", "frac_changed", "n_segments"),
) -> Dict[str, Any]:
    """
    per_seed_metrics: metrics computed for the SAME x but different seeds / runs.
    Returns variance and range for selected keys. Deterministic.
    """
    out: Dict[str, Any] = {"n_runs": len(per_seed_metrics)}
    for k in keys:
        vals = []
        for m in per_seed_metrics:
            v = _safe_float(m.get(k))
            if v is not None:
                vals.append(v)
        if not vals:
            out[k] = {"var": None, "std": None, "min": None, "max": None, "range": None}
            continue
        t = torch.tensor(vals, dtype=torch.float64)
        var = float(t.var(unbiased=False).item()) if t.numel() > 1 else 0.0
        std = float(torch.sqrt(torch.tensor(var)).item())
        mn = float(t.min().item())
        mx = float(t.max().item())
        out[k] = {"var": var, "std": std, "min": mn, "max": mx, "range": float(mx - mn)}
    return out
