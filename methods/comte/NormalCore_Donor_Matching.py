# Part 2 — NormalCore Donor Matching
# - extract donor segments from NormalCore
# - compute similarity using external libraries (DTW / Euclidean)
# - select top-k donor segments per candidate segment
#
# HARD RULES:
# - donor segment length == target segment length
# - contiguous temporal segments only
# - preserve cross-sensor coupling (no feature-wise matching)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw

Tensor = torch.Tensor


# -------------------------------
# Data containers
# -------------------------------


@dataclass(frozen=True)
class DonorSegment:
    donor_id: int  # index in NormalCore
    start: int
    end: int  # exclusive
    length: int
    distance: float  # similarity distance (lower is better)


@dataclass
class DonorMatchResult:
    segment: Tuple[int, int]
    donors: List[DonorSegment]


@dataclass
class DonorMatchConfig:
    metric: str = "dtw"  # {"dtw","euclidean"}
    standardize: bool = True
    max_donors: int = 5  # top-k donors per segment
    max_scan_per_donor: Optional[int] = None  # None = scan all possible positions


# -------------------------------
# Utilities
# -------------------------------


def _standardize_pair(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize x and y jointly to preserve relative scale.
    """
    Z = np.concatenate([x, y], axis=0)
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)
    return Zs[: len(x)], Zs[len(x) :]


def _distance(
    a: np.ndarray,
    b: np.ndarray,
    metric: str,
) -> float:
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if metric == "dtw":
        # tslearn DTW works on multivariate time series directly
        return float(dtw(a, b))
    raise ValueError(f"Unknown metric={metric}")


# -------------------------------
# Core matcher
# -------------------------------


class NormalCoreDonorMatcher:
    def __init__(
        self,
        normal_core: Tensor,  # (K,L,F)
        cfg: Optional[DonorMatchConfig] = None,
    ):
        if normal_core.ndim != 3:
            raise ValueError("NormalCore must have shape (K,L,F)")
        self.normal_core = normal_core.detach().float().cpu()
        self.cfg = cfg or DonorMatchConfig()

    def _extract_target(
        self,
        x: Tensor,
        seg: Tuple[int, int],
    ) -> np.ndarray:
        s, t = seg
        return x[s:t].detach().float().cpu().numpy()

    def _scan_donor(
        self,
        donor: np.ndarray,  # (L,F)
        target: np.ndarray,  # (l,F)
        donor_id: int,
    ) -> List[DonorSegment]:
        cfg = self.cfg
        L, F = donor.shape
        seg_len = target.shape[0]

        max_pos = L - seg_len
        if max_pos < 0:
            return []

        positions = range(max_pos + 1)
        if cfg.max_scan_per_donor is not None:
            positions = list(positions)[: cfg.max_scan_per_donor]

        matches: List[DonorSegment] = []

        for s in positions:
            t = s + seg_len
            cand = donor[s:t]

            if cfg.standardize:
                tgt_s, cand_s = _standardize_pair(target, cand)
            else:
                tgt_s, cand_s = target, cand

            d = _distance(tgt_s, cand_s, cfg.metric)

            matches.append(
                DonorSegment(
                    donor_id=donor_id,
                    start=s,
                    end=t,
                    length=seg_len,
                    distance=d,
                )
            )

        return matches

    def match(
        self,
        x: Tensor,  # (L,F)
        segment_candidates: List,  # from Part 1
    ) -> Dict[Tuple[int, int], DonorMatchResult]:
        """
        Returns mapping:
          (start,end) -> DonorMatchResult
        """
        results: Dict[Tuple[int, int], DonorMatchResult] = {}
        cfg = self.cfg
        total_segments = len(segment_candidates)
        for i, seg in enumerate(segment_candidates):
            # <--- Add Progress Print --->
            print(
                f"  [Matcher] Processing segment {i + 1}/{total_segments} (len={seg.length})..."
            )

            key = (seg.start, seg.end)
            target = self._extract_target(x, key)
            all_matches = []

            # This inner loop is the heavy part
            for k in range(self.normal_core.shape[0]):
                # <--- Add Granular Print (only if really slow) --->
                if k % 100 == 0:
                    print(
                        f"    Scanning donor {k}/{self.normal_core.shape[0]}", end="\r"
                    )

                donor_ts = self.normal_core[k].numpy()
                matches = self._scan_donor(
                    donor=donor_ts,
                    target=target,
                    donor_id=k,
                )
                all_matches.extend(matches)

            if not all_matches:
                continue

            # rank globally by similarity
            all_matches.sort(key=lambda z: z.distance)
            topk = all_matches[: cfg.max_donors]

            results[key] = DonorMatchResult(
                segment=key,
                donors=topk,
            )

        return results


# -------------------------------
# Example minimal usage (no I/O)
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # fake data
    K, L, F = 10, 128, 6
    normal_core = torch.randn(K, L, F)
    x = torch.randn(L, F)

    # fake segment candidates (normally from Part 1)
    class _Seg:
        def __init__(self, s, t):
            self.start = s
            self.end = t

    segments = [_Seg(20, 36), _Seg(60, 84)]

    matcher = NormalCoreDonorMatcher(
        normal_core,
        DonorMatchConfig(
            metric="dtw",
            standardize=True,
            max_donors=3,
        ),
    )

    donor_matches = matcher.match(x, segments)

    # donor_matches[(start,end)].donors -> ranked donor segments
    for k, v in donor_matches.items():
        assert len(v.donors) <= 3
