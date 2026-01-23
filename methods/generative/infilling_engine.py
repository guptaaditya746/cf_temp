# part2_infilling_engine.py
# Generative Infilling Counterfactuals — Part 2: Infilling Engine
#
# Responsibilities:
# - apply masks from Part 1 (time or time_feature masks)
# - construct x_masked with a configurable mask token
# - call a generative infiller model (library-first: torch)
# - produce multiple stochastic candidate fills per mask (seeded)
# - NEVER modify unmasked regions
#
# Notes:
# - This module does NOT evaluate anomaly scores or constraints (Part 3).
# - It returns candidates + metadata so downstream can validate/rank.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import torch
from einops import rearrange

# re-use MaskSpec definitions (import from part1 if in a package)
# from part1_mask_strategy import MaskSpec
# For standalone use, we duck-type MaskSpec (needs mask_type, segments, feature_indices, schedule_id, meta).


# ----------------------------
# Infiller Model Interface
# ----------------------------
class InfillerModel(Protocol):
    """
    Minimal callable interface for an infilling model.
    You can wrap any pre-trained imputer (Transformer, RNN AE, diffusion imputer, etc.)
    as long as it satisfies this interface.

    Contract:
    - input: x_masked (B,L,F), mask (B,L,F) uint8/bool indicating masked cells
    - output: x_filled (B,L,F)
    - must not change unmasked values (engine will enforce overwrite safety anyway)
    """

    def __call__(self, x_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...


# ----------------------------
# Config
# ----------------------------
@dataclass
class InfillingConfig:
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    # mask token strategies
    mask_token_mode: str = "zeros"  # "zeros" | "feature_mean" | "custom"
    custom_mask_token_value: float = 0.0

    # sampling
    n_samples: int = 8
    sampling_seed: int = 0
    deterministic: bool = (
        False  # if True, forces same output each sample (if model supports)
    )

    # safety / post-process
    clamp_output: bool = False
    value_min: Optional[float] = None
    value_max: Optional[float] = None

    # optional smoothing only within masked region (kept light; realism checks happen in Part 3)
    smooth_masked: bool = False
    smooth_sigma: float = (
        0.0  # if >0, gaussian smooth along time axis for masked cells only
    )


# ----------------------------
# Outputs
# ----------------------------
@dataclass
class InfillCandidate:
    x_cf: torch.Tensor  # (L,F)
    mask: torch.Tensor  # (L,F) bool
    masked_segment: Tuple[int, int]  # (min_start, max_end) over segments in spec
    mask_size: int
    sample_id: int
    seed: int
    schedule_id: str
    meta: Dict[str, Any]


# ----------------------------
# Engine
# ----------------------------
class InfillingEngine:
    def __init__(
        self,
        infiller: InfillerModel,
        cfg: InfillingConfig,
        normalcore_feature_mean: Optional[np.ndarray] = None,  # (F,)
    ):
        self.infiller = infiller
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.normalcore_feature_mean = None
        if normalcore_feature_mean is not None:
            arr = np.asarray(normalcore_feature_mean).astype(np.float32)
            if arr.ndim != 1:
                raise ValueError(
                    f"normalcore_feature_mean must be (F,), got {arr.shape}"
                )
            self.normalcore_feature_mean = torch.from_numpy(arr).to(
                self.device, dtype=cfg.dtype
            )

        self._base_rng = np.random.default_rng(cfg.sampling_seed)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,  # (L,F)
        mask_spec: Any,  # MaskSpec-like
        mask_builder,  # callable: (L,F,spec) -> np.ndarray (L,) or (L,F)
        immutable_features: Optional[Iterable[int]] = None,
    ) -> List[InfillCandidate]:
        """
        Generate multiple infilled candidates for one mask_spec.
        - x must be (L,F) torch tensor.
        - mask_builder should be MaskStrategy.build_mask from Part 1.
        """
        x = self._ensure_x(x)
        L, F = x.shape
        imm = (
            set(int(i) for i in immutable_features)
            if immutable_features is not None
            else set()
        )

        # Build a (L,F) boolean mask from spec (time mask expands to all non-immutable features)
        M_np = mask_builder(L=L, F=F, spec=mask_spec)
        M = self._normalize_mask(M_np, L=L, F=F, imm=imm).to(self.device)

        # Construct masked input
        x_masked = self._apply_mask_token(x, M, F)

        # Batch sampling
        B = int(max(1, self.cfg.n_samples))
        seeds = self._derive_seeds(B)

        x_batch = x_masked.unsqueeze(0).expand(B, -1, -1).contiguous()
        m_batch = M.unsqueeze(0).expand(B, -1, -1).contiguous()

        candidates: List[InfillCandidate] = []

        for b in range(B):
            # set per-sample seed for reproducibility (best-effort; models may ignore)
            self._seed_everything(int(seeds[b]))

            # model forward
            x_filled_b = self.infiller(
                x_batch[b : b + 1], m_batch[b : b + 1]
            )  # (1,L,F)
            if not isinstance(x_filled_b, torch.Tensor):
                raise TypeError("InfillerModel must return a torch.Tensor")
            if x_filled_b.ndim != 3 or x_filled_b.shape != (1, L, F):
                raise ValueError(
                    f"Infiller output must be (1,L,F) got {tuple(x_filled_b.shape)}"
                )

            x_filled_b = x_filled_b[0]

            # STRICT: keep unmasked region identical to original x
            x_cf = torch.where(M, x_filled_b, x)

            # optional post-processing inside masked region only
            if self.cfg.smooth_masked and float(self.cfg.smooth_sigma) > 0.0:
                x_cf = self._smooth_only_masked(
                    x_cf, M, sigma=float(self.cfg.smooth_sigma)
                )

            if self.cfg.clamp_output:
                x_cf = self._clamp(x_cf)

            seg_span = self._segment_span(mask_spec, L)
            mask_size = int(M.sum().item())

            meta = {
                "mask_token_mode": self.cfg.mask_token_mode,
                "smooth_masked": bool(self.cfg.smooth_masked),
                "smooth_sigma": float(self.cfg.smooth_sigma),
                "deterministic": bool(self.cfg.deterministic),
                "libraries_used": ["torch", "numpy", "einops"],
                "schedule_meta": getattr(mask_spec, "meta", {}),
                "feature_indices": getattr(mask_spec, "feature_indices", None),
                "mask_type": getattr(mask_spec, "mask_type", "unknown"),
            }

            candidates.append(
                InfillCandidate(
                    x_cf=x_cf.detach().clone(),
                    mask=M.detach().clone(),
                    masked_segment=seg_span,
                    mask_size=mask_size,
                    sample_id=b,
                    seed=int(seeds[b]),
                    schedule_id=str(getattr(mask_spec, "schedule_id", "unknown")),
                    meta=meta,
                )
            )

        return candidates

    # ----------------------------
    # Mask + token handling
    # ----------------------------
    def _normalize_mask(
        self, M_np: np.ndarray, L: int, F: int, imm: set
    ) -> torch.Tensor:
        """
        Normalize mask into (L,F) boolean tensor.
        - time mask: (L,) -> expand to all features except immutable
        - time_feature mask: (L,F) -> enforce immutables off
        """
        M_np = np.asarray(M_np)
        if M_np.ndim == 1:
            if M_np.shape[0] != L:
                raise ValueError(f"Time mask must be (L,), got {M_np.shape}")
            M2 = np.repeat(M_np[:, None], F, axis=1)
        elif M_np.ndim == 2:
            if M_np.shape != (L, F):
                raise ValueError(f"Time-feature mask must be (L,F), got {M_np.shape}")
            M2 = M_np
        else:
            raise ValueError(f"Mask must be 1D or 2D, got {M_np.shape}")

        # immutable features cannot be masked
        if imm:
            imm_idx = [i for i in imm if 0 <= i < F]
            if imm_idx:
                M2[:, imm_idx] = 0

        M = torch.from_numpy(M2.astype(np.bool_, copy=False))
        return M

    def _apply_mask_token(
        self, x: torch.Tensor, M: torch.Tensor, F: int
    ) -> torch.Tensor:
        mode = str(self.cfg.mask_token_mode).lower()

        if mode == "zeros":
            token = torch.zeros((1, 1, F), device=self.device, dtype=self.cfg.dtype)
        elif mode == "feature_mean":
            if self.normalcore_feature_mean is None:
                raise ValueError(
                    "mask_token_mode='feature_mean' requires normalcore_feature_mean"
                )
            token = self.normalcore_feature_mean.view(1, 1, F).to(
                self.device, dtype=self.cfg.dtype
            )
        elif mode == "custom":
            token = torch.full(
                (1, 1, F),
                float(self.cfg.custom_mask_token_value),
                device=self.device,
                dtype=self.cfg.dtype,
            )
        else:
            raise ValueError(f"Unknown mask_token_mode: {self.cfg.mask_token_mode}")

        # x_masked = x*(1-M) + token*M
        x_masked = torch.where(M, token.view(1, F).expand_as(x), x)
        return x_masked

    # ----------------------------
    # Utilities
    # ----------------------------
    def _ensure_x(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 2:
            raise ValueError(f"x must be (L,F), got {tuple(x.shape)}")
        x = x.to(self.device, dtype=self.cfg.dtype)
        return x

    def _derive_seeds(self, B: int) -> np.ndarray:
        # stable but distinct per sample
        base = int(self.cfg.sampling_seed)
        if self.cfg.deterministic:
            return np.full((B,), base, dtype=np.int64)
        # draw B seeds deterministically from base_rng
        return self._base_rng.integers(low=0, high=2**31 - 1, size=(B,), dtype=np.int64)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        # Best-effort; some models may use their own RNG streams.
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        vmin = self.cfg.value_min
        vmax = self.cfg.value_max
        if vmin is None and vmax is None:
            return x
        if vmin is None:
            return torch.clamp(x, max=float(vmax))
        if vmax is None:
            return torch.clamp(x, min=float(vmin))
        return torch.clamp(x, min=float(vmin), max=float(vmax))

    @staticmethod
    def _segment_span(mask_spec: Any, L: int) -> Tuple[int, int]:
        segs = getattr(mask_spec, "segments", None)
        if not segs:
            return (0, 0)
        starts = [max(0, min(int(s.start), L)) for s in segs]
        ends = [max(0, min(int(s.end), L)) for s in segs]
        return (int(min(starts)), int(max(ends)))

    @staticmethod
    def _smooth_only_masked(
        x_cf: torch.Tensor, M: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """
        Light gaussian smoothing along time, applied ONLY on masked positions.
        Implemented in torch using conv1d per feature for speed and no extra deps.
        """
        if sigma <= 0.0:
            return x_cf

        # Build 1D gaussian kernel
        # radius ~ 3*sigma
        radius = int(max(1, round(3.0 * sigma)))
        t = torch.arange(-radius, radius + 1, device=x_cf.device, dtype=x_cf.dtype)
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)
        kernel = kernel / kernel.sum()
        k = kernel.view(1, 1, -1)

        L, F = x_cf.shape
        # (F,1,L)
        x_f = rearrange(x_cf, "l f -> f 1 l")
        m_f = rearrange(M.to(x_cf.dtype), "l f -> f 1 l")

        # smooth both x and mask weights to avoid leaking unmasked values
        x_num = torch.nn.functional.conv1d(x_f * m_f, k, padding=radius)
        x_den = torch.nn.functional.conv1d(m_f, k, padding=radius).clamp_min(1e-6)
        x_smooth = x_num / x_den

        x_smooth = rearrange(x_smooth, "f 1 l -> l f")

        # replace only masked cells
        return torch.where(M, x_smooth, x_cf)


# ----------------------------
# Example wrapper infiller (baseline)
# ----------------------------
class SimpleDenoiseInfiller:
    """
    Baseline infiller to get the pipeline running:
    - fills masked region by local linear interpolation per feature
    - then optionally smooths (engine can do extra smoothing)
    This is NOT meant to be SOTA, but it's a valid black-box infiller component.

    Replace with your actual pre-trained infiller model wrapper.
    """

    def __call__(self, x_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x_masked, mask: (B,L,F)
        B, L, F = x_masked.shape
        out = x_masked.clone()

        # For each sample/feature: interpolate masked timesteps using nearest unmasked neighbors
        for b in range(B):
            for f in range(F):
                m = mask[b, :, f].bool()
                if not torch.any(m):
                    continue
                y = out[b, :, f]
                idx = torch.arange(L, device=out.device)

                known = ~m
                if torch.sum(known) < 2:
                    # if almost everything masked, fallback to zeros (keeps it simple)
                    y[m] = 0.0
                    out[b, :, f] = y
                    continue

                x_known = idx[known].to(out.dtype)
                y_known = y[known]

                # linear interpolation: for each masked i, find surrounding known points
                x_all = idx.to(out.dtype)

                # Use torch searchsorted by converting to cpu if needed (still library-first)
                xk = x_known
                # ensure sorted (it is)
                # For each x, find right insertion point
                right = torch.searchsorted(xk, x_all).clamp(1, xk.numel() - 1)
                left = right - 1
                x0, x1 = xk[left], xk[right]
                y0, y1 = y_known[left], y_known[right]
                w = (x_all - x0) / (x1 - x0).clamp_min(1e-6)
                y_interp = y0 + w * (y1 - y0)

                y[m] = y_interp[m]
                out[b, :, f] = y

        return out


# ----------------------------
# Minimal self-test
# ----------------------------
if __name__ == "__main__":
    # Dummy inputs
    L, F = 64, 4
    x = torch.randn(L, F)

    # Dummy mask_spec-like object
    class _Seg:
        def __init__(self, s, e):
            self.start, self.end = s, e

    class _Spec:
        mask_type = "time"
        segments = (_Seg(20, 30),)
        feature_indices = None
        schedule_id = "demo:seg0:base"
        meta = {"base_len": 10.0}

    # Dummy builder: time mask (L,)
    def _builder(L: int, F: int, spec: Any) -> np.ndarray:
        M = np.zeros((L,), dtype=np.uint8)
        for seg in spec.segments:
            M[int(seg.start) : int(seg.end)] = 1
        return M

    infiller = SimpleDenoiseInfiller()
    cfg = InfillingConfig(
        device="cpu",
        mask_token_mode="zeros",
        n_samples=3,
        sampling_seed=123,
        smooth_masked=True,
        smooth_sigma=1.0,
    )
    engine = InfillingEngine(
        infiller=infiller,
        cfg=cfg,
        normalcore_feature_mean=np.zeros((F,), dtype=np.float32),
    )

    cands = engine.generate(
        x=x, mask_spec=_Spec(), mask_builder=_builder, immutable_features=[3]
    )
    print("Candidates:", len(cands))
    print(
        "First cand masked size:",
        cands[0].mask_size,
        "segment span:",
        cands[0].masked_segment,
    )
