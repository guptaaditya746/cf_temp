# Part 1 — Segment Candidate Generation (CoMTE-style, reconstruction-based)
# - identify anomalous segments (from reconstruction error signal)
# - define candidate segment lengths (fixed + multi-resolution)
# - rank candidate segments (by error mass / peakness / compactness)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from einops import reduce
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

Tensor = torch.Tensor


@dataclass(frozen=True)
class SegmentCandidate:
    start: int
    end: int  # exclusive
    length: int
    score: float  # higher = more anomalous segment candidate
    peak_t: int
    peak_value: float
    mass: float
    mean: float
    maxv: float


@dataclass(frozen=True)
class SegmentGenConfig:
    # reconstruction-error signal construction
    per_feature_error: str = "l1"  # {"l1","l2"}
    feature_reduce: str = "mean"  # {"mean","sum","max"}
    feature_reduce: str = "mean"  # {"mean","sum","max"}
    smooth_sigma: float = 2.0  # gaussian smoothing over time (0 disables)

    # peak detection
    peak_prominence: float = 0.0  # set >0 to suppress tiny peaks
    peak_distance: int = 1  # min distance between peaks (in timesteps)
    max_peaks: int = 12  # cap number of peaks to expand around

    # segment lengths
    lengths: Tuple[int, ...] = (8, 12, 16, 24, 32, 48)
    allow_multi_resolution: bool = True
    min_len: int = 4
    max_len: Optional[int] = None  # if None, max is L

    # candidate pruning
    topk_per_peak: int = 24
    global_topk: int = 200
    dedup_iou: float = 0.90  # deduplicate near-identical segments

    # scoring weights (ranking)
    w_mass: float = 1.0
    w_peak: float = 0.25
    w_compact: float = 0.25  # penalize long segments if same mass


def _to_1d_np(x: Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    return x.contiguous().view(-1).numpy()


def _safe_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))


def _segment_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter0 = max(a0, b0)
    inter1 = min(a1, b1)
    inter = max(0, inter1 - inter0)
    union = (a1 - a0) + (b1 - b0) - inter
    return 0.0 if union <= 0 else inter / union


def _reduce_features(err_t_f: Tensor, mode: str) -> Tensor:
    # err_t_f: (L,F)
    if mode == "mean":
        return err_t_f.mean(dim=-1)
    if mode == "sum":
        return err_t_f.sum(dim=-1)
    if mode == "max":
        return err_t_f.max(dim=-1).values
    raise ValueError(f"Unknown feature_reduce={mode}")


def _per_feature_error(x: Tensor, x_hat: Tensor, mode: str) -> Tensor:
    # returns (L,F)
    if mode == "l1":
        return (x - x_hat).abs()
    if mode == "l2":
        return (x - x_hat).pow(2)
    raise ValueError(f"Unknown per_feature_error={mode}")


def compute_recon_error_signal(
    x: Tensor,
    reconstructor: Callable[[Tensor], Tensor],
    cfg: SegmentGenConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    x: (L,F) torch
    reconstructor: callable returning x_hat (L,F) torch
    Returns:
      e_smooth: (L,) np
      aux: dict with raw & per-feature stats (np arrays)
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (L,F), got shape={tuple(x.shape)}")

    with torch.no_grad():
        x_hat = reconstructor(x)
    if x_hat.shape != x.shape:
        raise ValueError(
            f"x_hat shape must match x. got x_hat={tuple(x_hat.shape)} x={tuple(x.shape)}"
        )

    err_t_f = _per_feature_error(x, x_hat, cfg.per_feature_error)  # (L,F)
    err_t = _reduce_features(err_t_f, cfg.feature_reduce)  # (L,)

    e_raw = _to_1d_np(err_t)
    e = e_raw.copy()

    if cfg.smooth_sigma and cfg.smooth_sigma > 0:
        e = gaussian_filter1d(e, sigma=float(cfg.smooth_sigma))

    aux = {
        "e_raw": e_raw,
        "e_smooth": e,
        "e_feat_mean": _to_1d_np(err_t_f.mean(dim=-1)),
        "e_feat_max": _to_1d_np(err_t_f.max(dim=-1).values),
    }
    return e, aux


def detect_peaks(
    e: np.ndarray,
    cfg: SegmentGenConfig,
) -> np.ndarray:
    """
    e: (L,) smoothed error signal
    returns peak indices sorted by descending peak height (then by time)
    """
    if e.ndim != 1:
        raise ValueError("e must be 1D")

    peaks, props = find_peaks(
        e,
        prominence=float(cfg.peak_prominence)
        if cfg.peak_prominence is not None
        else None,
        distance=int(cfg.peak_distance),
    )
    if peaks.size == 0:
        return peaks

    heights = e[peaks]
    order = np.lexsort((-heights, peaks))  # desc by height, asc by time
    peaks = peaks[order]

    if cfg.max_peaks and peaks.size > cfg.max_peaks:
        peaks = peaks[: cfg.max_peaks]
    return peaks


def _candidate_lengths(L: int, cfg: SegmentGenConfig) -> List[int]:
    max_len = L if cfg.max_len is None else min(L, int(cfg.max_len))
    min_len = max(1, int(cfg.min_len))

    base = [int(l) for l in cfg.lengths if min_len <= int(l) <= max_len]
    if not cfg.allow_multi_resolution:
        return sorted(set(base))

    # add multi-res if not already: powers-of-two-ish between min_len and max_len
    ms = []
    p = 1
    while p < max_len * 2:
        p *= 2
        if min_len <= p <= max_len:
            ms.append(p)
    return sorted(set(base + ms))


def _segment_stats(e: np.ndarray, s: int, t: int) -> Tuple[float, float, float]:
    seg = e[s:t]
    if seg.size == 0:
        return 0.0, 0.0, 0.0
    mass = float(seg.sum())
    mean = float(seg.mean())
    maxv = float(seg.max())
    return mass, mean, maxv


def expand_segments_around_peaks(
    e: np.ndarray,
    peaks: np.ndarray,
    cfg: SegmentGenConfig,
) -> List[SegmentCandidate]:
    """
    Generate candidate segments by centering/aligning around peaks with multiple lengths.
    """
    L = int(e.shape[0])
    lens = _candidate_lengths(L, cfg)
    candidates: List[SegmentCandidate] = []

    for p in peaks.tolist():
        p = int(p)
        peak_val = float(e[p])

        # for each length, try a few alignments around peak (left/center/right)
        for seg_len in lens:
            seg_len = int(seg_len)
            if seg_len <= 0 or seg_len > L:
                continue

            # alignments: peak at 25%, 50%, 75% of the segment
            for frac in (0.25, 0.50, 0.75):
                s = int(round(p - frac * seg_len))
                s = _safe_int(s, 0, L - seg_len)
                t = s + seg_len

                mass, mean, maxv = _segment_stats(e, s, t)

                # compactness: prefer shorter if same mass
                compact = mass / float(seg_len)

                # ranking score: error mass + peakness + compactness
                score = (
                    cfg.w_mass * mass + cfg.w_peak * peak_val + cfg.w_compact * compact
                )
                candidates.append(
                    SegmentCandidate(
                        start=s,
                        end=t,
                        length=seg_len,
                        score=float(score),
                        peak_t=p,
                        peak_value=float(peak_val),
                        mass=float(mass),
                        mean=float(mean),
                        maxv=float(maxv),
                    )
                )

    # per-peak top-k prune
    if cfg.topk_per_peak and cfg.topk_per_peak > 0:
        pruned: List[SegmentCandidate] = []
        by_peak: Dict[int, List[SegmentCandidate]] = {}
        for c in candidates:
            by_peak.setdefault(c.peak_t, []).append(c)
        for p, lst in by_peak.items():
            lst_sorted = sorted(lst, key=lambda z: z.score, reverse=True)
            pruned.extend(lst_sorted[: cfg.topk_per_peak])
        candidates = pruned

    return candidates


def dedup_segments(
    candidates: List[SegmentCandidate],
    cfg: SegmentGenConfig,
) -> List[SegmentCandidate]:
    """
    Greedy dedup by IoU, keeping higher-score segment.
    """
    if not candidates:
        return []

    cand_sorted = sorted(candidates, key=lambda z: z.score, reverse=True)
    kept: List[SegmentCandidate] = []
    for c in cand_sorted:
        ok = True
        for k in kept:
            if _segment_iou((c.start, c.end), (k.start, k.end)) >= float(cfg.dedup_iou):
                ok = False
                break
        if ok:
            kept.append(c)

    return kept


def rank_and_select(
    candidates: List[SegmentCandidate],
    cfg: SegmentGenConfig,
) -> List[SegmentCandidate]:
    if not candidates:
        return []
    c_sorted = sorted(candidates, key=lambda z: z.score, reverse=True)
    if cfg.global_topk and cfg.global_topk > 0:
        c_sorted = c_sorted[: cfg.global_topk]
    return c_sorted


@dataclass
class SegmentCandidateOutput:
    candidates: List[SegmentCandidate]
    signals: Dict[str, np.ndarray]  # e_raw, e_smooth, ...
    config: Dict


class SegmentCandidateGenerator:
    def __init__(self, cfg: Optional[SegmentGenConfig] = None):
        self.cfg = cfg or SegmentGenConfig()

    def generate(
        self,
        x: Tensor,
        reconstructor: Callable[[Tensor], Tensor],
    ) -> SegmentCandidateOutput:
        """
        x: (L,F) torch
        reconstructor: callable x->x_hat (L,F)
        """
        cfg = self.cfg
        e, aux = compute_recon_error_signal(x, reconstructor, cfg)
        peaks = detect_peaks(e, cfg)

        # Fallback if no peaks: treat global max as a single peak
        if peaks.size == 0:
            p = int(np.argmax(e)) if e.size > 0 else 0
            peaks = np.asarray([p], dtype=int)

        candidates = expand_segments_around_peaks(e, peaks, cfg)
        candidates = dedup_segments(candidates, cfg)
        candidates = rank_and_select(candidates, cfg)

        signals = dict(aux)
        signals["peaks"] = peaks.astype(int)

        return SegmentCandidateOutput(
            candidates=candidates,
            signals=signals,
            config=asdict(cfg),
        )


# -------------------------------
# Example minimal usage (no I/O)
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    L, F = 128, 6
    x = torch.randn(L, F)

    # dummy reconstructor: slightly smoothed version
    def reconstructor(inp: Tensor) -> Tensor:
        # cheap temporal smoothing using conv1d
        y = inp.T.unsqueeze(0)  # (1,F,L)
        k = torch.ones(1, 1, 5, device=inp.device, dtype=inp.dtype) / 5.0
        y2 = torch.nn.functional.conv1d(y, k.repeat(F, 1, 1), padding=2, groups=F)
        return y2.squeeze(0).T  # (L,F)

    gen = SegmentCandidateGenerator(
        SegmentGenConfig(
            smooth_sigma=2.0,
            peak_prominence=0.0,
            peak_distance=4,
            lengths=(8, 12, 16, 24, 32),
            allow_multi_resolution=True,
            topk_per_peak=20,
            global_topk=120,
        )
    )
    out = gen.generate(x, reconstructor)

    # out.candidates: ranked segments to try substituting in Part 3
    # out.signals: error signal + peaks for debugging/visualization
    assert isinstance(out.candidates, list)
