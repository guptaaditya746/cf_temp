# part1_mask_strategy.py
# Generative Infilling Counterfactuals — Part 1: Mask Strategy
#
# Responsibilities:
# - take per-timestep reconstruction error e[t] (and optionally per-feature error e[t,f])
# - extract contiguous anomalous segments (top-k, width constraints)
# - build masks M (L) or (L,F) for contiguous edits only
# - provide deterministic mask schedules: expand/contract, beam candidates, multi-segment options
#
# Library-first: numpy, scipy, sklearn (optional), dataclasses

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class Segment:
    start: int  # inclusive
    end: int  # exclusive

    @property
    def length(self) -> int:
        return int(self.end - self.start)

    def clamp(self, L: int) -> "Segment":
        s = int(max(0, min(self.start, L)))
        e = int(max(0, min(self.end, L)))
        if e < s:
            e = s
        return Segment(s, e)

    def expand(self, left: int, right: int, L: int) -> "Segment":
        return Segment(self.start - int(left), self.end + int(right)).clamp(L)

    def intersect(self, other: "Segment") -> Optional["Segment"]:
        s = max(self.start, other.start)
        e = min(self.end, other.end)
        if e <= s:
            return None
        return Segment(s, e)

    def overlaps(self, other: "Segment") -> bool:
        return not (self.end <= other.start or other.end <= self.start)


@dataclass(frozen=True)
class MaskSpec:
    mask_type: str  # "time" (L,) or "time_feature" (L,F)
    segments: Tuple[Segment, ...]
    feature_indices: Optional[Tuple[int, ...]]  # None means all features
    schedule_id: str
    meta: Dict[str, float]


@dataclass
class MaskStrategyConfig:
    # segment extraction
    top_k: int = 3
    min_len: int = 4
    max_len: int = 64
    merge_gap: int = 2

    # smoothing for e[t] to avoid spiky singletons
    smooth_sigma: float = 1.0
    robust_quantile: float = 0.90  # threshold = quantile(e_smooth)

    # candidate generation
    max_candidates: int = 48
    expand_steps: Tuple[int, ...] = (0, 2, 4, 6, 8)  # symmetric expansion sizes
    edge_expand_bias: bool = True  # if True, also try left-only/right-only expansions

    # feature selection (optional)
    per_feature_mode: bool = False  # if True, create masks for selected features only
    top_feat_k: int = (
        3  # how many features to include if per_feature_mode and e_tf provided
    )

    # safety
    random_seed: int = 0


class MaskStrategy:
    """
    Generates contiguous masks targeting high-error segments.
    Does NOT do any infilling; just proposes what to mask.
    """

    def __init__(self, cfg: MaskStrategyConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

    # ----------------------------
    # Public API
    # ----------------------------
    def propose(
        self,
        e_t: np.ndarray,  # shape (L,)
        L: Optional[int] = None,
        e_tf: Optional[np.ndarray] = None,  # shape (L,F) optional
        immutable_features: Optional[Iterable[int]] = None,
    ) -> List[MaskSpec]:
        """
        Returns a list of MaskSpec candidates sorted roughly from minimal -> larger edits.

        Inputs:
        - e_t: per-timestep reconstruction error (L,)
        - e_tf: per-timestep-per-feature error (L,F) if available (enables feature masks)
        - immutable_features: indices that must never be masked (for time_feature masks)
        """
        e_t = self._as_1d_float(e_t)
        L_ = int(L) if L is not None else int(e_t.shape[0])
        if e_t.shape[0] != L_:
            raise ValueError(f"e_t length {e_t.shape[0]} does not match L={L_}")

        imm = (
            set(int(i) for i in immutable_features)
            if immutable_features is not None
            else set()
        )

        # 1) Extract base segments from robust thresholding on smoothed error
        e_s = self._smooth_error(e_t, sigma=self.cfg.smooth_sigma)
        base_segments = self._extract_segments(
            e_s,
            top_k=self.cfg.top_k,
            min_len=self.cfg.min_len,
            max_len=self.cfg.max_len,
            merge_gap=self.cfg.merge_gap,
            thr=float(np.quantile(e_s, self.cfg.robust_quantile)),
        )

        if not base_segments:
            return []

        # 2) Convert segments into a candidate set with expansions/contractions
        candidates: List[MaskSpec] = []
        schedule_id_prefix = "mask_v1"

        # Feature selection (optional, only if e_tf provided and enabled)
        feat_sets: List[Optional[Tuple[int, ...]]] = [None]
        if self.cfg.per_feature_mode and (e_tf is not None):
            e_tf = self._as_2d_float(e_tf)
            if e_tf.shape[0] != L_:
                raise ValueError(
                    f"e_tf first dim {e_tf.shape[0]} does not match L={L_}"
                )
            feat_sets = self._select_feature_sets(e_tf, base_segments, imm)

        # 3) Build mask schedules
        for seg_i, seg in enumerate(base_segments):
            for feat_idx_tuple in feat_sets:
                # base
                candidates.append(
                    MaskSpec(
                        mask_type="time_feature"
                        if feat_idx_tuple is not None
                        else "time",
                        segments=(seg,),
                        feature_indices=feat_idx_tuple,
                        schedule_id=f"{schedule_id_prefix}:seg{seg_i}:base",
                        meta={"base_len": float(seg.length)},
                    )
                )

                # expansions (symmetric + optionally one-sided)
                for step in self.cfg.expand_steps:
                    if step <= 0:
                        continue

                    # symmetric
                    s_sym = seg.expand(step, step, L_)
                    candidates.append(
                        MaskSpec(
                            mask_type="time_feature"
                            if feat_idx_tuple is not None
                            else "time",
                            segments=(s_sym,),
                            feature_indices=feat_idx_tuple,
                            schedule_id=f"{schedule_id_prefix}:seg{seg_i}:exp{step}x2",
                            meta={
                                "base_len": float(seg.length),
                                "expanded_len": float(s_sym.length),
                                "expand_left": float(step),
                                "expand_right": float(step),
                            },
                        )
                    )

                    if self.cfg.edge_expand_bias:
                        s_l = seg.expand(step, 0, L_)
                        s_r = seg.expand(0, step, L_)
                        candidates.append(
                            MaskSpec(
                                mask_type="time_feature"
                                if feat_idx_tuple is not None
                                else "time",
                                segments=(s_l,),
                                feature_indices=feat_idx_tuple,
                                schedule_id=f"{schedule_id_prefix}:seg{seg_i}:left{step}",
                                meta={
                                    "base_len": float(seg.length),
                                    "expanded_len": float(s_l.length),
                                    "expand_left": float(step),
                                    "expand_right": 0.0,
                                },
                            )
                        )
                        candidates.append(
                            MaskSpec(
                                mask_type="time_feature"
                                if feat_idx_tuple is not None
                                else "time",
                                segments=(s_r,),
                                feature_indices=feat_idx_tuple,
                                schedule_id=f"{schedule_id_prefix}:seg{seg_i}:right{step}",
                                meta={
                                    "base_len": float(seg.length),
                                    "expanded_len": float(s_r.length),
                                    "expand_left": 0.0,
                                    "expand_right": float(step),
                                },
                            )
                        )

        # 4) Optional multi-segment masks (top-2, top-3 combined), still contiguous per segment
        # This helps when anomalies occur in separated bursts; still no scattered point edits.
        if len(base_segments) > 1:
            combos = self._segment_combinations(
                base_segments, max_combo=min(3, len(base_segments))
            )
            for c_i, segs in enumerate(combos):
                # skip singletons (already added)
                if len(segs) <= 1:
                    continue
                for feat_idx_tuple in feat_sets:
                    candidates.append(
                        MaskSpec(
                            mask_type="time_feature"
                            if feat_idx_tuple is not None
                            else "time",
                            segments=tuple(segs),
                            feature_indices=feat_idx_tuple,
                            schedule_id=f"{schedule_id_prefix}:combo{c_i}:n{len(segs)}",
                            meta={
                                "combo_total_len": float(sum(s.length for s in segs))
                            },
                        )
                    )

        # 5) Deduplicate and sort minimal-first
        candidates = self._dedup(candidates)
        candidates.sort(key=lambda ms: (self._mask_size_proxy(ms), len(ms.segments)))

        if len(candidates) > self.cfg.max_candidates:
            candidates = candidates[: self.cfg.max_candidates]

        return candidates

    def build_mask(
        self,
        L: int,
        F: int,
        spec: MaskSpec,
    ) -> np.ndarray:
        """
        Create a binary mask array from a MaskSpec.
        - time mask: shape (L,) with 1 on masked timesteps
        - time_feature mask: shape (L,F) with 1 on masked cells, only for selected features
        """
        L = int(L)
        F = int(F)
        if spec.mask_type == "time":
            M = np.zeros((L,), dtype=np.uint8)
            for seg in spec.segments:
                s = seg.clamp(L)
                if s.end > s.start:
                    M[s.start : s.end] = 1
            return M

        if spec.mask_type == "time_feature":
            M = np.zeros((L, F), dtype=np.uint8)
            feats = spec.feature_indices
            if feats is None:
                feats = tuple(range(F))
            for seg in spec.segments:
                s = seg.clamp(L)
                if s.end > s.start:
                    M[s.start : s.end, list(feats)] = 1
            return M

        raise ValueError(f"Unknown mask_type: {spec.mask_type}")

    # ----------------------------
    # Internals
    # ----------------------------
    @staticmethod
    def _as_1d_float(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {x.shape}")
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _as_2d_float(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _smooth_error(e_t: np.ndarray, sigma: float) -> np.ndarray:
        if sigma is None or float(sigma) <= 0.0:
            return e_t
        return gaussian_filter1d(e_t, sigma=float(sigma), mode="nearest")

    def _extract_segments(
        self,
        e_s: np.ndarray,
        top_k: int,
        min_len: int,
        max_len: int,
        merge_gap: int,
        thr: float,
    ) -> List[Segment]:
        """
        Extract contiguous segments where e_s >= thr, then:
        - merge close segments within merge_gap
        - filter by min_len
        - cap by max_len (split long segments)
        - pick top_k by segment 'mass' (sum of e_s in segment)
        """
        L = int(e_s.shape[0])
        min_len = int(max(1, min_len))
        max_len = int(max(min_len, max_len))
        merge_gap = int(max(0, merge_gap))
        top_k = int(max(1, top_k))

        # boolean exceedance
        on = (e_s >= float(thr)).astype(np.uint8)

        # contiguous runs
        segs: List[Segment] = []
        i = 0
        while i < L:
            if on[i] == 0:
                i += 1
                continue
            j = i + 1
            while j < L and on[j] == 1:
                j += 1
            segs.append(Segment(i, j))
            i = j

        if not segs:
            # fallback: force a segment around global max (robust when everything is below quantile due to flat errors)
            t_max = int(np.argmax(e_s))
            half = max(1, min_len // 2)
            segs = [Segment(t_max - half, t_max + half + 1).clamp(L)]

        # merge close ones
        segs = self._merge_close(segs, gap=merge_gap)

        # split overly long segments to respect max_len (keep highest-energy subwindows)
        segs = self._split_long(segs, e_s, max_len=max_len)

        # filter min_len
        segs = [s for s in segs if s.length >= min_len]

        if not segs:
            return []

        # rank by energy (sum)
        energies = [float(np.sum(e_s[s.start : s.end])) for s in segs]
        order = np.argsort(-np.asarray(energies))
        segs = [segs[int(idx)] for idx in order[:top_k]]

        # final sort by time
        segs.sort(key=lambda s: (s.start, s.end))
        return segs

    @staticmethod
    def _merge_close(segs: List[Segment], gap: int) -> List[Segment]:
        if not segs:
            return []
        gap = int(max(0, gap))
        segs_sorted = sorted(segs, key=lambda s: (s.start, s.end))
        merged: List[Segment] = [segs_sorted[0]]
        for s in segs_sorted[1:]:
            last = merged[-1]
            if s.start <= last.end + gap:
                merged[-1] = Segment(last.start, max(last.end, s.end))
            else:
                merged.append(s)
        return merged

    def _split_long(
        self, segs: List[Segment], e_s: np.ndarray, max_len: int
    ) -> List[Segment]:
        out: List[Segment] = []
        max_len = int(max(1, max_len))
        for s in segs:
            if s.length <= max_len:
                out.append(s)
                continue

            # sliding window over the segment, pick multiple subwindows greedily by energy, non-overlapping
            seg_err = e_s[s.start : s.end]
            win = max_len
            # compute window sums efficiently
            csum = np.cumsum(np.concatenate([[0.0], seg_err.astype(np.float32)]))
            w_sums = csum[win:] - csum[:-win]  # length = len(seg_err)-win+1
            if w_sums.size <= 0:
                out.append(Segment(s.start, s.start + max_len))
                continue

            used = np.zeros_like(w_sums, dtype=np.uint8)
            # pick up to ceil(len/ max_len) windows
            picks = int(np.ceil(s.length / max_len))
            for _ in range(picks):
                # mask out used overlaps by setting to -inf
                masked = np.where(used == 1, -np.inf, w_sums)
                idx = int(np.argmax(masked))
                if not np.isfinite(masked[idx]):
                    break
                a = idx
                b = idx + win
                out.append(Segment(s.start + a, s.start + b))
                # mark overlaps in w_sums indices (any window that overlaps [a,b))
                lo = max(0, a - win + 1)
                hi = min(w_sums.size, b)  # exclusive
                used[lo:hi] = 1

        # merge close again (splits might touch)
        out = self._merge_close(out, gap=0)
        return out

    def _select_feature_sets(
        self,
        e_tf: np.ndarray,  # (L,F)
        base_segments: List[Segment],
        immutable: set,
    ) -> List[Optional[Tuple[int, ...]]]:
        """
        Choose a few feature index sets to try masking, based on aggregate error over base segments.
        Ensures immutable features are excluded.
        Returns list of tuples of feature indices (sorted).
        """
        L, F = e_tf.shape
        agg = np.zeros((F,), dtype=np.float32)
        for seg in base_segments:
            s = seg.clamp(L)
            if s.end > s.start:
                agg += np.sum(e_tf[s.start : s.end, :], axis=0)

        # exclude immutable
        for f in immutable:
            if 0 <= f < F:
                agg[f] = -np.inf

        # pick top-k features
        k = int(max(1, min(self.cfg.top_feat_k, F)))
        top = np.argsort(-agg)[:k]
        top = [int(i) for i in top if np.isfinite(agg[int(i)])]

        if not top:
            return [None]

        # return a few sets: individual features + combined set
        sets: List[Tuple[int, ...]] = [(i,) for i in top]
        sets.append(tuple(sorted(top)))
        # dedup
        uniq: List[Tuple[int, ...]] = []
        seen = set()
        for s in sets:
            if s not in seen:
                seen.add(s)
                uniq.append(tuple(sorted(s)))
        return uniq

    @staticmethod
    def _segment_combinations(
        segs: List[Segment], max_combo: int
    ) -> List[List[Segment]]:
        """
        Simple combinations: take first n segments in time order, for n=1..max_combo.
        Practical and predictable (no random subset explosion).
        """
        segs_sorted = sorted(segs, key=lambda s: (s.start, s.end))
        combos: List[List[Segment]] = []
        for n in range(1, int(max_combo) + 1):
            combos.append(segs_sorted[:n])
        return combos

    @staticmethod
    def _mask_size_proxy(spec: MaskSpec) -> int:
        """
        Proxy for sorting before we know F:
        - time masks: total timesteps masked
        - time_feature masks: timesteps * num_features_selected (approx)
        """
        t = sum(s.length for s in spec.segments)
        if spec.mask_type == "time":
            return int(t)
        feats = spec.feature_indices
        f = len(feats) if feats is not None else 9999  # push "all-features" later
        return int(t) * int(f)

    @staticmethod
    def _dedup(cands: List[MaskSpec]) -> List[MaskSpec]:
        """
        Deduplicate by (mask_type, segments, feature_indices).
        Keep first occurrence (already roughly minimal-first).
        """
        seen = set()
        out: List[MaskSpec] = []
        for c in cands:
            key = (
                c.mask_type,
                tuple((s.start, s.end) for s in c.segments),
                c.feature_indices,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out


# ----------------------------
# Minimal self-test / usage
# ----------------------------
if __name__ == "__main__":
    L, F = 128, 5
    # toy error: two anomalous bursts
    e_t = np.random.rand(L).astype(np.float32) * 0.2
    e_t[30:40] += 1.5
    e_t[80:88] += 1.2

    e_tf = np.abs(np.random.randn(L, F).astype(np.float32)) * 0.05
    e_tf[30:40, 2] += 1.0
    e_tf[80:88, 1] += 0.8

    cfg = MaskStrategyConfig(
        top_k=3,
        min_len=4,
        max_len=24,
        smooth_sigma=1.0,
        robust_quantile=0.90,
        per_feature_mode=True,
        top_feat_k=2,
        max_candidates=32,
        random_seed=0,
    )
    ms = MaskStrategy(cfg)

    specs = ms.propose(e_t=e_t, L=L, e_tf=e_tf, immutable_features=[4])
    print(f"Generated {len(specs)} mask specs")
    for i, s in enumerate(specs[:10]):
        print(i, s.schedule_id, s.mask_type, s.segments, s.feature_indices, s.meta)

    # Build a mask for the first spec
    if specs:
        M = ms.build_mask(L=L, F=F, spec=specs[0])
        print("Mask shape:", M.shape, "masked count:", int(M.sum()))
