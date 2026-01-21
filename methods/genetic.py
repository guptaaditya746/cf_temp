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


class SegmentGeneticBlendCF(BaseCounterfactual):
    def __init__(
        self,
        model,
        threshold: float,
        device: str | torch.device,
        normal_core: torch.Tensor,
        seed: int = 0,
        max_evals: int = 4000,
        min_seg_len: int = 4,
        max_seg_len: int = 64,
        pop_size: int = 32,
        generations: int = 40,
        elite_frac: float = 0.25,
        mutation_prob: float = 0.35,
        alpha_grid: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.5, 0.65, 0.8),
        lam_dist: float = 0.35,
        lam_smooth: float = 0.2,
        bounds_q: Tuple[float, float] = (0.01, 0.99),
        require_margin: float = 0.0,
    ):
        super().__init__(model, threshold, device)
        self.normal_core = normal_core
        self.seed = int(seed)
        self.max_evals = int(max_evals)
        self.min_seg_len = int(min_seg_len)
        self.max_seg_len = int(max_seg_len)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.elite_frac = float(elite_frac)
        self.mutation_prob = float(mutation_prob)
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.lam_dist = float(lam_dist)
        self.lam_smooth = float(lam_smooth)
        self.require_margin = float(require_margin)
        self._bounds = _compute_feature_bounds(
            self.normal_core, bounds_q[0], bounds_q[1], self.device
        )

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(x, self._bounds.hi), self._bounds.lo)

    def _smooth_penalty(
        self, x: torch.Tensor, x0: torch.Tensor, s: int, e: int
    ) -> float:
        if e - s <= 1:
            return 0.0
        dx = x[s:e] - x0[s:e]
        d1 = dx[1:] - dx[:-1]
        return float(torch.mean(d1**2).detach().cpu().item())

    def _fitness(
        self, x: torch.Tensor, x0: torch.Tensor, sc: float, s: int, e: int
    ) -> float:
        dist = float(_l2(x, x0).detach().cpu().item())
        sm = self._smooth_penalty(x, x0, s, e)
        penalty = max(0.0, sc - (self.threshold - self.require_margin))
        return float(penalty + self.lam_dist * dist + self.lam_smooth * sm)

    def generate(self, x: torch.Tensor) -> Optional[Dict[str, Any]]:
        x0 = _to_2d(x).to(self.device)
        L, F = x0.shape
        g = torch.Generator(device=self.device).manual_seed(self.seed)

        s0 = self.score(x0)
        if s0 <= (self.threshold - self.require_margin):
            return {
                "x_cf": x0.detach().clone(),
                "score": float(s0),
                "meta": {"already_valid": True, "evals": 1},
            }

        segs = _segment_choices(L, self.min_seg_len, self.max_seg_len)
        if not segs:
            return None

        nc = self.normal_core.to(self.device)
        K = nc.shape[0]
        evals = 1

        def sample_gene() -> Tuple[int, int, int, float]:
            seg_idx = int(
                torch.randint(
                    0, len(segs), (1,), generator=g, device=self.device
                ).item()
            )
            s, e = segs[seg_idx]
            p = int(torch.randint(0, K, (1,), generator=g, device=self.device).item())
            a_idx = int(
                torch.randint(
                    0, len(self.alpha_grid), (1,), generator=g, device=self.device
                ).item()
            )
            a = self.alpha_grid[a_idx]
            return (s, e, p, a)

        def express(gene: Tuple[int, int, int, float]) -> torch.Tensor:
            s, e, p, a = gene
            x1 = x0.clone()
            x1[s:e] = (1.0 - a) * x1[s:e] + a * nc[p, s:e]
            return self._clip(x1)

        pop = [sample_gene() for _ in range(self.pop_size)]

        best_x = None
        best_score = float("inf")
        best_dist = float("inf")
        best_gene = None

        elite_n = max(1, int(round(self.elite_frac * self.pop_size)))

        for gen in range(self.generations):
            if evals >= self.max_evals:
                break

            scored: List[Tuple[float, float, Tuple[int, int, int, float], float]] = []
            for gene in pop:
                if evals >= self.max_evals:
                    break
                x1 = express(gene)
                sc = self.score(x1)
                evals += 1
                fit = self._fitness(x1, x0, sc, gene[0], gene[1])
                dist = float(_l2(x0, x1).detach().cpu().item())
                scored.append((fit, sc, gene, dist))
                if sc <= (self.threshold - self.require_margin):
                    if (sc < best_score - 1e-12) or (
                        abs(sc - best_score) <= 1e-12 and dist < best_dist
                    ):
                        best_x = x1.clone()
                        best_score = float(sc)
                        best_dist = dist
                        best_gene = gene

            if not scored:
                break

            scored.sort(key=lambda t: t[0])
            elites = [t[2] for t in scored[:elite_n]]

            def crossover(
                a: Tuple[int, int, int, float], b: Tuple[int, int, int, float]
            ) -> Tuple[int, int, int, float]:
                s1, e1, p1, al1 = a
                s2, e2, p2, al2 = b
                pick = int(
                    torch.randint(0, 4, (1,), generator=g, device=self.device).item()
                )
                if pick == 0:
                    return (s1, e1, p2, al2)
                if pick == 1:
                    return (s2, e2, p1, al1)
                if pick == 2:
                    return (s1, e1, p1, al2)
                return (s2, e2, p2, al1)

            def mutate(
                gene: Tuple[int, int, int, float],
            ) -> Tuple[int, int, int, float]:
                s, e, p, a = gene
                if (
                    float(torch.rand((1,), generator=g, device=self.device).item())
                    > self.mutation_prob
                ):
                    return gene
                mtype = int(
                    torch.randint(0, 3, (1,), generator=g, device=self.device).item()
                )
                if mtype == 0:
                    seg_idx = int(
                        torch.randint(
                            0, len(segs), (1,), generator=g, device=self.device
                        ).item()
                    )
                    s, e = segs[seg_idx]
                elif mtype == 1:
                    p = int(
                        torch.randint(
                            0, K, (1,), generator=g, device=self.device
                        ).item()
                    )
                else:
                    a_idx = int(
                        torch.randint(
                            0,
                            len(self.alpha_grid),
                            (1,),
                            generator=g,
                            device=self.device,
                        ).item()
                    )
                    a = self.alpha_grid[a_idx]
                return (s, e, p, a)

            new_pop: List[Tuple[int, int, int, float]] = []
            new_pop.extend(elites)
            while len(new_pop) < self.pop_size:
                i = int(
                    torch.randint(
                        0, len(elites), (1,), generator=g, device=self.device
                    ).item()
                )
                j = int(
                    torch.randint(
                        0, len(elites), (1,), generator=g, device=self.device
                    ).item()
                )
                child = crossover(elites[i], elites[j])
                child = mutate(child)
                new_pop.append(child)
            pop = new_pop

        if best_x is None:
            return None

        s, e, p, a = best_gene if best_gene is not None else (0, 0, 0, 0.0)
        return {
            "x_cf": best_x.detach().clone(),
            "score": float(best_score),
            "meta": {
                "evals": evals,
                "distance_l2": float(best_dist),
                "gene": {
                    "segment": (int(s), int(e)),
                    "prototype": int(p),
                    "alpha": float(a),
                },
                "used_normal_core": True,
                "require_margin": float(self.require_margin),
            },
        }
