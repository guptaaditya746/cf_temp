

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch


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
            if hasattr(self.model, "reconstruction_score") and callable(getattr(self.model, "reconstruction_score")):
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


def _compute_feature_bounds(normal_core: torch.Tensor, q_lo: float, q_hi: float, device: torch.device) -> _Bounds:
    nc = normal_core.to(device)
    if nc.dim() != 3:
        raise ValueError(f"NormalCore must have shape (K,L,F), got {tuple(nc.shape)}")
    flat = nc.reshape(-1, nc.shape[-1])
    lo = torch.quantile(flat, q_lo, dim=0)
    hi = torch.quantile(flat, q_hi, dim=0)
    eps = 1e-6
    hi = torch.maximum(hi, lo + eps)
    return _Bounds(lo=lo, hi=hi)

class SegmentSimulatedAnnealingCF(BaseCounterfactual):
    def __init__(
        self,
        model,
        threshold: float,
        device: str | torch.device,
        normal_core: torch.Tensor,
        seed: int = 0,
        max_evals: int = 2000,
        min_seg_len: int = 4,
        max_seg_len: int = 64,
        steps: int = 600,
        t0: float = 1.0,
        t_min: float = 0.02,
        lam_score: float = 1.0,
        lam_dist: float = 0.25,
        lam_smooth: float = 0.15,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
        bounds_q: Tuple[float, float] = (0.01, 0.99),
        require_margin: float = 0.0,
    ):
        super().__init__(model, threshold, device)
        self.normal_core = normal_core
        self.seed = int(seed)
        self.max_evals = int(max_evals)
        self.min_seg_len = int(min_seg_len)
        self.max_seg_len = int(max_seg_len)
        self.steps = int(steps)
        self.t0 = float(t0)
        self.t_min = float(t_min)
        self.lam_score = float(lam_score)
        self.lam_dist = float(lam_dist)
        self.lam_smooth = float(lam_smooth)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.require_margin = float(require_margin)
        self._bounds = _compute_feature_bounds(self.normal_core, bounds_q[0], bounds_q[1], self.device)

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(x, self._bounds.hi), self._bounds.lo)

    def _smooth_penalty(self, x: torch.Tensor, x0: torch.Tensor, s: int, e: int) -> torch.Tensor:
        if e - s <= 1:
            return torch.tensor(0.0, device=self.device)
        dx = (x[s:e] - x0[s:e])
        d1 = dx[1:] - dx[:-1]
        return torch.mean(d1 ** 2)

    def _objective(self, x: torch.Tensor, x0: torch.Tensor, s: int, e: int) -> float:
        sc = self.score(x)
        dist = _l2(x, x0)
        sm = self._smooth_penalty(x, x0, s, e)
        val = self.lam_score * max(0.0, sc - (self.threshold - self.require_margin)) + self.lam_dist * float(dist.detach().cpu().item()) + self.lam_smooth * float(sm.detach().cpu().item())
        return float(val)

    def generate(self, x: torch.Tensor) -> Optional[Dict[str, Any]]:
        x0 = _to_2d(x).to(self.device)
        L, F = x0.shape
        g = torch.Generator(device=self.device).manual_seed(self.seed)

        s0 = self.score(x0)
        if s0 <= (self.threshold - self.require_margin):
            return {"x_cf": x0.detach().clone(), "score": float(s0), "meta": {"already_valid": True, "evals": 1}}

        segs = _segment_choices(L, self.min_seg_len, self.max_seg_len)
        if not segs:
            return None

        nc = self.normal_core.to(self.device)
        K = nc.shape[0]
        evals = 1

        init_seg_idx = int(torch.randint(0, len(segs), (1,), generator=g, device=self.device).item())
        s, e = segs[init_seg_idx]
        proto_idx = int(torch.randint(0, K, (1,), generator=g, device=self.device).item())
        alpha = float(torch.rand((1,), generator=g, device=self.device).item()) * (self.alpha_max - self.alpha_min) + self.alpha_min

        x_cur = x0.clone()
        x_cur[s:e] = (1.0 - alpha) * x_cur[s:e] + alpha * nc[proto_idx, s:e]
        x_cur = self._clip(x_cur)

        obj_cur = self._objective(x_cur, x0, s, e)

        best_x = None
        best_score = float("inf")
        best_dist = float("inf")

        for step in range(self.steps):
            if evals >= self.max_evals:
                break

            frac = step / max(1, self.steps - 1)
            T = max(self.t_min, self.t0 * (1.0 - frac))

            move_type = int(torch.randint(0, 3, (1,), generator=g, device=self.device).item())

            s2, e2 = s, e
            proto2 = proto_idx
            alpha2 = alpha

            if move_type == 0:
                seg_idx2 = int(torch.randint(0, len(segs), (1,), generator=g, device=self.device).item())
                s2, e2 = segs[seg_idx2]
            elif move_type == 1:
                proto2 = int(torch.randint(0, K, (1,), generator=g, device=self.device).item())
            else:
                jitter = (torch.rand((1,), generator=g, device=self.device).item() - 0.5) * 0.2
                alpha2 = float(min(self.alpha_max, max(self.alpha_min, alpha2 + jitter)))

            x_prop = x0.clone()
            x_prop[s2:e2] = (1.0 - alpha2) * x_prop[s2:e2] + alpha2 * nc[proto2, s2:e2]
            x_prop = self._clip(x_prop)

            obj_prop = self._objective(x_prop, x0, s2, e2)
            evals += 1

            accept = False
            if obj_prop <= obj_cur:
                accept = True
            else:
                p = torch.exp(torch.tensor(-(obj_prop - obj_cur) / max(1e-12, T), device=self.device))
                u = torch.rand((1,), generator=g, device=self.device)
                accept = bool((u <= p).item())

            if accept:
                x_cur = x_prop
                obj_cur = obj_prop
                s, e = s2, e2
                proto_idx = proto2
                alpha = alpha2

            sc_cur = self.score(x_cur)
            evals += 1
            if sc_cur <= (self.threshold - self.require_margin):
                dist = float(_l2(x0, x_cur).detach().cpu().item())
                if (sc_cur < best_score - 1e-12) or (abs(sc_cur - best_score) <= 1e-12 and dist < best_dist):
                    best_x = x_cur.clone()
                    best_score = float(sc_cur)
                    best_dist = dist

        if best_x is None:
            return None

        return {
            "x_cf": best_x.detach().clone(),
            "score": float(best_score),
            "meta": {
                "evals": evals,
                "distance_l2": float(best_dist),
                "last_segment": (int(s), int(e)),
                "last_alpha": float(alpha),
                "used_normal_core": True,
                "require_margin": float(self.require_margin),
            },
        }

