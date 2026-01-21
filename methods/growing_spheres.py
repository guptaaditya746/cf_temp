from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from methods.base import BaseCounterfactual


def _as_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected x with shape (L,F) or (B,L,F), got {tuple(x.shape)}")


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError(f"Expected x with shape (L,F) or (1,L,F), got {tuple(x.shape)}")


def _segment_choices(L: int, min_len: int, max_len: int) -> List[Tuple[int, int]]:
    max_len = max(min_len, min(max_len, L))
    min_len = max(1, min(min_len, L))
    choices: List[Tuple[int, int]] = []
    for seg_len in range(min_len, max_len + 1):
        for start in range(0, L - seg_len + 1):
            choices.append((start, start + seg_len))
    return choices


def _l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x - y) ** 2) + 1e-12)


def _contiguous_mask(L: int, s: int, e: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros((L,), dtype=torch.bool, device=device)
    m[s:e] = True
    return m


class BaseCounterfactual:
    def __init__(self, model, threshold: float, device: str | torch.device):
        self.model = model
        self.threshold = float(threshold)
        self.device = torch.device(device)

    def score(self, x: torch.Tensor) -> float:
        x3 = _as_3d(x).to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "reconstruction_score") and callable(
                getattr(self.model, "reconstruction_score")
            ):
                s = self.model.reconstruction_score(x3)
            else:
                out = self.model(x3)
                if isinstance(out, (tuple, list)) and len(out) > 0:
                    s = out[-1]
                else:
                    s = out
            if isinstance(s, torch.Tensor):
                if s.numel() != 1:
                    s = s.reshape(-1)[0]
                return float(s.detach().cpu().item())
            return float(s)

    def generate(self, x: torch.Tensor) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


@dataclass(frozen=True)
class _Bounds:
    lo: torch.Tensor  # (F,)
    hi: torch.Tensor  # (F,)


def _compute_feature_bounds(
    normal_core: torch.Tensor, q_lo: float, q_hi: float, device: torch.device
) -> _Bounds:
    nc = normal_core.to(device)
    if nc.dim() != 3:
        raise ValueError(f"NormalCore must have shape (K,L,F), got {tuple(nc.shape)}")
    flat = nc.reshape(-1, nc.shape[-1])
    lo = torch.quantile(flat, q_lo, dim=0)
    hi = torch.quantile(flat, q_hi, dim=0)
    eps = 1e-6
    hi = torch.maximum(hi, lo + eps)
    return _Bounds(lo=lo, hi=hi)


class SegmentGrowingSpheresCF(BaseCounterfactual):
    def __init__(
        self,
        model,
        threshold: float,
        device: str | torch.device,
        normal_core: Optional[torch.Tensor] = None,
        seed: int = 0,
        max_evals: int = 1500,
        min_seg_len: int = 4,
        max_seg_len: int = 64,
        n_dirs: int = 16,
        n_radii: int = 24,
        r_min: float = 0.01,
        r_max: float = 3.0,
        refine_steps: int = 10,
        bounds_q: Tuple[float, float] = (0.01, 0.99),
    ):
        super().__init__(model, threshold, device)
        self.normal_core = normal_core
        self.seed = int(seed)
        self.max_evals = int(max_evals)
        self.min_seg_len = int(min_seg_len)
        self.max_seg_len = int(max_seg_len)
        self.n_dirs = int(n_dirs)
        self.n_radii = int(n_radii)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.refine_steps = int(refine_steps)
        self.bounds_q = bounds_q
        self._bounds: Optional[_Bounds] = None
        if self.normal_core is not None:
            self._bounds = _compute_feature_bounds(
                self.normal_core, bounds_q[0], bounds_q[1], self.device
            )

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        if self._bounds is None:
            return x
        return torch.max(torch.min(x, self._bounds.hi), self._bounds.lo)

    def generate(self, x: torch.Tensor) -> Optional[Dict[str, Any]]:
        x0 = _to_2d(x).to(self.device)
        L, F = x0.shape
        g = torch.Generator(device=self.device).manual_seed(self.seed)

        s0 = self.score(x0)
        if s0 <= self.threshold:
            return {
                "x_cf": x0.detach().clone(),
                "score": float(s0),
                "meta": {"already_valid": True, "evals": 1},
            }

        segs = _segment_choices(L, self.min_seg_len, self.max_seg_len)
        if not segs:
            return None

        radii = torch.logspace(
            torch.log10(torch.tensor(self.r_min, device=self.device)),
            torch.log10(torch.tensor(self.r_max, device=self.device)),
            steps=self.n_radii,
        )

        best_x = None
        best_score = float("inf")
        best_dist = float("inf")
        evals = 1

        for s, e in segs:
            seg_len = e - s
            mask_t = _contiguous_mask(L, s, e, self.device).unsqueeze(-1)  # (L,1)
            for _ in range(self.n_dirs):
                d = torch.randn((seg_len, F), generator=g, device=self.device)
                d = d / (torch.norm(d) + 1e-12)
                for r in radii:
                    if evals >= self.max_evals:
                        if best_x is None:
                            return None
                        return {
                            "x_cf": best_x.detach().clone(),
                            "score": float(best_score),
                            "meta": {
                                "evals": evals,
                                "segment": (int(s), int(e)),
                                "early_stop": True,
                            },
                        }

                    x1 = x0.clone()
                    x1[s:e] = x1[s:e] + (r * d)
                    x1 = self._clip(x1)

                    sc = self.score(x1)
                    evals += 1
                    if sc <= self.threshold:
                        lo_r = 0.0
                        hi_r = float(r.detach().cpu().item())
                        x_best_local = x1
                        sc_best_local = sc
                        for _rs in range(self.refine_steps):
                            mid = 0.5 * (lo_r + hi_r)
                            x_mid = x0.clone()
                            x_mid[s:e] = x_mid[s:e] + (mid * d)
                            x_mid = self._clip(x_mid)
                            sc_mid = self.score(x_mid)
                            evals += 1
                            if sc_mid <= self.threshold:
                                hi_r = mid
                                x_best_local = x_mid
                                sc_best_local = sc_mid
                            else:
                                lo_r = mid

                        dist = float(_l2(x0, x_best_local).detach().cpu().item())
                        if (sc_best_local < best_score - 1e-12) or (
                            abs(sc_best_local - best_score) <= 1e-12
                            and dist < best_dist
                        ):
                            best_x = x_best_local
                            best_score = float(sc_best_local)
                            best_dist = dist
                        break

        if best_x is None:
            return None

        return {
            "x_cf": best_x.detach().clone(),
            "score": float(best_score),
            "meta": {
                "evals": evals,
                "distance_l2": float(best_dist),
                "used_normal_core_bounds": self._bounds is not None,
            },
        }
