from typing import Any, Dict, Optional, Tuple

import torch


class MotifGuidedSegmentRepairCF(BaseCounterfactual):
    def __init__(
        self,
        model,
        threshold: float,
        device: str,
        normal_core: torch.Tensor,  # (K, L, F)
        *,
        max_segments_per_len: int = 12,
        top_motifs_per_segment: int = 8,
        lengths: Tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20, 24),
        edge_blend: int = 2,
        value_clip_quantiles: Tuple[float, float] = (0.01, 0.99),
        use_error_guidance: bool = True,
    ):
        super().__init__(model, threshold, device)
        if normal_core is None:
            raise ValueError("normal_core is required")
        if normal_core.dim() != 3:
            raise ValueError("normal_core must have shape (K,L,F)")
        self.normal_core = normal_core.to(device)
        self.max_segments_per_len = int(max_segments_per_len)
        self.top_motifs_per_segment = int(top_motifs_per_segment)
        self.lengths = tuple(int(x) for x in lengths)
        self.edge_blend = int(edge_blend)
        self.value_clip_quantiles = value_clip_quantiles
        self.use_error_guidance = bool(use_error_guidance)

        qlo, qhi = self.value_clip_quantiles
        flat = self.normal_core.reshape(-1, self.normal_core.shape[-1])
        self._clip_lo = torch.quantile(flat, qlo, dim=0)
        self._clip_hi = torch.quantile(flat, qhi, dim=0)

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(x, self._clip_hi), self._clip_lo)

    def _recon_and_per_t_error(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.model.reconstruct(x)
        per_t = ((x - x_hat) ** 2).mean(dim=-1)  # (L,)
        return x_hat, per_t

    def _candidate_starts(
        self, per_t: torch.Tensor, seg_len: int, L: int
    ) -> torch.Tensor:
        if L - seg_len <= 0:
            return torch.empty(0, dtype=torch.long, device=per_t.device)

        win = per_t.unfold(0, seg_len, 1).mean(dim=-1)  # (L-seg_len+1,)
        n = min(self.max_segments_per_len, win.numel())
        if n <= 0:
            return torch.empty(0, dtype=torch.long, device=per_t.device)

        idx = torch.topk(win, k=n, largest=True).indices
        idx, _ = torch.sort(idx)
        return idx.to(torch.long)

    def _all_motifs_of_len(self, seg_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        K, L, F = self.normal_core.shape
        if seg_len > L:
            return torch.empty(0, seg_len, F, device=self.device), torch.empty(
                0, 2, dtype=torch.long, device=self.device
            )

        motifs = []
        meta = []
        for k in range(K):
            w = self.normal_core[k]  # (L,F)
            for s in range(0, L - seg_len + 1):
                motifs.append(w[s : s + seg_len])
                meta.append((k, s))
        if not motifs:
            return torch.empty(0, seg_len, F, device=self.device), torch.empty(
                0, 2, dtype=torch.long, device=self.device
            )

        motifs_t = torch.stack(motifs, dim=0)  # (M, seg_len, F)
        meta_t = torch.tensor(meta, dtype=torch.long, device=self.device)  # (M,2)
        return motifs_t, meta_t

    def _top_motifs(self, x_seg: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
        # x_seg: (seg_len,F), motifs: (M,seg_len,F)
        if motifs.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        diff = motifs - x_seg.unsqueeze(0)
        d = (diff * diff).mean(dim=(1, 2))  # (M,)
        k = min(self.top_motifs_per_segment, d.numel())
        return torch.topk(d, k=k, largest=False).indices

    def _apply_substitution(
        self,
        x: torch.Tensor,
        start: int,
        motif: torch.Tensor,
        edge_blend: int,
    ) -> torch.Tensor:
        # x: (L,F), motif: (seg_len,F)
        L, F = x.shape
        seg_len = motif.shape[0]
        out = x.clone()

        s = int(start)
        e = s + seg_len

        if edge_blend <= 0 or seg_len <= 2 * edge_blend:
            out[s:e] = motif
            return out

        b = edge_blend
        out[s + b : e - b] = motif[b : seg_len - b]

        # left blend
        for i in range(b):
            t = i / max(1, b)
            out[s + i] = (1.0 - t) * x[s + i] + t * motif[i]

        # right blend
        for i in range(b):
            t = i / max(1, b)
            out[e - b + i] = (1.0 - t) * motif[seg_len - b + i] + t * x[e - b + i]

        return out

    def generate(self, x: torch.Tensor) -> Optional[Dict[str, Any]]:
        if x.dim() != 2:
            raise ValueError("x must have shape (L,F)")
        x = x.to(self.device)
        L, F = x.shape

        base_score = (
            float(self.score(x).item())
            if torch.is_tensor(self.score(x))
            else float(self.score(x))
        )
        if base_score <= self.threshold:
            return {
                "x_cf": x.clone(),
                "score": float(base_score),
                "meta": {"already_valid": True},
            }

        if self.use_error_guidance:
            _, per_t = self._recon_and_per_t_error(x)
        else:
            per_t = torch.zeros(L, device=self.device)

        best = None
        best_key = None

        for seg_len in self.lengths:
            if seg_len <= 0 or seg_len > L:
                continue

            motifs, motifs_meta = self._all_motifs_of_len(seg_len)
            if motifs.numel() == 0:
                continue

            if self.use_error_guidance:
                starts = self._candidate_starts(per_t, seg_len, L)
                if starts.numel() == 0:
                    starts = torch.arange(
                        0, L - seg_len + 1, device=self.device, dtype=torch.long
                    )
            else:
                starts = torch.arange(
                    0, L - seg_len + 1, device=self.device, dtype=torch.long
                )

            for s in starts.tolist():
                x_seg = x[s : s + seg_len]
                top_idx = self._top_motifs(x_seg, motifs)
                if top_idx.numel() == 0:
                    continue

                for mi in top_idx.tolist():
                    motif = motifs[mi]
                    x_cf = self._apply_substitution(x, s, motif, self.edge_blend)
                    x_cf = self._clip(x_cf)

                    sc = (
                        float(self.score(x_cf).item())
                        if torch.is_tensor(self.score(x_cf))
                        else float(self.score(x_cf))
                    )
                    if sc > self.threshold:
                        continue

                    delta = x_cf - x
                    l2 = float(torch.norm(delta).item())
                    l1 = float(delta.abs().sum().item())

                    key = (seg_len, l1, l2, sc)
                    if best is None or key < best_key:
                        k_id, k_start = (
                            int(motifs_meta[mi, 0].item()),
                            int(motifs_meta[mi, 1].item()),
                        )
                        best = {
                            "x_cf": x_cf,
                            "score": float(sc),
                            "meta": {
                                "method": "motif_guided_segment_repair",
                                "base_score": float(base_score),
                                "threshold": float(self.threshold),
                                "segment_start": int(s),
                                "segment_len": int(seg_len),
                                "motif_source_k": k_id,
                                "motif_source_start": k_start,
                                "edge_blend": int(self.edge_blend),
                                "l1_change": float(l1),
                                "l2_change": float(l2),
                                "clip_quantiles": tuple(self.value_clip_quantiles),
                                "used_error_guidance": bool(self.use_error_guidance),
                            },
                        }
                        best_key = key

            if best is not None:
                return best

        return None
