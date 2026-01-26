# Part 3 — Segment Substitution Engine
# - apply contiguous segment replacement
# - enforce boundary smoothness
# - preserve immutable sensors
#
# HARD RULES:
# - replace exactly one contiguous segment
# - no masking, no interpolation across the whole window
# - cross-sensor structure preserved (copy full F-dim vectors)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal.windows import tukey

Tensor = torch.Tensor


# -------------------------------
# Config & containers
# -------------------------------


@dataclass
class SubstitutionConfig:
    boundary_smoothing: bool = True
    smoothing_alpha: float = 0.5  # Tukey window alpha (0=no smoothing)
    immutable_sensors: Optional[List[int]] = None  # feature indices
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None


@dataclass
class SubstitutionResult:
    x_cf: Tensor
    replaced_segment: Tuple[int, int]
    donor_id: int
    donor_segment: Tuple[int, int]
    boundary_applied: bool


# -------------------------------
# Utilities
# -------------------------------


def _apply_bounds(x: Tensor, lo: Optional[float], hi: Optional[float]) -> Tensor:
    if lo is not None:
        x = torch.maximum(x, torch.tensor(lo, device=x.device, dtype=x.dtype))
    if hi is not None:
        x = torch.minimum(x, torch.tensor(hi, device=x.device, dtype=x.dtype))
    return x


def _blend_boundary(
    original: Tensor,
    donor: Tensor,
    alpha: float,
) -> Tensor:
    """
    Smooth only at the boundaries using a Tukey window.
    donor: (l,F)
    """
    if alpha <= 0:
        return donor

    l = donor.shape[0]
    w = tukey(l, alpha=alpha)
    w = torch.from_numpy(w).to(donor.device, donor.dtype).unsqueeze(-1)
    return w * donor + (1.0 - w) * original


# -------------------------------
# Core substitution engine
# -------------------------------


class SegmentSubstitutor:
    def __init__(self, cfg: Optional[SubstitutionConfig] = None):
        self.cfg = cfg or SubstitutionConfig()

    def substitute(
        self,
        x: Tensor,  # (L,F)
        segment: Tuple[int, int],  # (s,t)
        donor_segment: Tensor,  # (l,F)
        donor_id: int,
        donor_range: Tuple[int, int],
    ) -> SubstitutionResult:
        """
        Returns a single counterfactual candidate.
        """
        if x.ndim != 2:
            raise ValueError("x must be (L,F)")

        s, t = segment
        l = t - s
        if donor_segment.shape[0] != l:
            raise ValueError("Donor length mismatch")

        x_cf = x.clone()

        # Preserve immutable sensors
        imm = set(self.cfg.immutable_sensors or [])

        # Original segment for blending reference
        orig_seg = x[s:t]

        # Prepare donor copy
        donor_seg = donor_segment.clone()

        if imm:
            for f in imm:
                donor_seg[:, f] = orig_seg[:, f]

        # Optional boundary smoothing
        boundary_applied = False
        if self.cfg.boundary_smoothing and self.cfg.smoothing_alpha > 0:
            donor_seg = _blend_boundary(
                original=orig_seg,
                donor=donor_seg,
                alpha=self.cfg.smoothing_alpha,
            )
            boundary_applied = True

        # Insert segment
        x_cf[s:t] = donor_seg

        # Optional global clamping
        x_cf = _apply_bounds(
            x_cf,
            lo=self.cfg.clamp_min,
            hi=self.cfg.clamp_max,
        )

        return SubstitutionResult(
            x_cf=x_cf,
            replaced_segment=(s, t),
            donor_id=donor_id,
            donor_segment=donor_range,
            boundary_applied=boundary_applied,
        )


# -------------------------------
# Example minimal usage (no I/O)
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    L, F = 128, 6
    x = torch.randn(L, F)

    # fake donor
    donor = torch.randn(16, F)

    substitutor = SegmentSubstitutor(
        SubstitutionConfig(
            boundary_smoothing=True,
            smoothing_alpha=0.4,
            immutable_sensors=[0],  # e.g. timestamp / ID sensor
            clamp_min=-5.0,
            clamp_max=5.0,
        )
    )

    res = substitutor.substitute(
        x=x,
        segment=(40, 56),
        donor_segment=donor,
        donor_id=3,
        donor_range=(10, 26),
    )

    assert res.x_cf.shape == x.shape
