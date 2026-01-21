import torch
from torch import Tensor
from typing import Optional, Dict, Any

class SegmentSubstitutionCounterfactual:
    """
    Gradient-free, localized counterfactual for time series.

    Strategy:
    - Replace contiguous segments with segments from normal prototypes
    - Keep original context unchanged
    - Return first minimal valid substitution
    """

    def __init__(
        self,
        model,
        threshold: float,
        normal_windows: Tensor,  # (K, L, F)
        segment_length: int,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.device = device or torch.device("cpu")

        assert normal_windows.ndim == 3, "normal_windows must be (K,L,F)"
        self.prototypes = normal_windows.to(self.device)

        self.segment_length = segment_length
        self.model.eval()

    @torch.no_grad()
    def _score(self, x: Tensor) -> float:
        x_hat = self.model(x.unsqueeze(0))
        score = ((x - x_hat.squeeze(0)) ** 2).mean()
        return float(score.item())

    @torch.no_grad()
    def generate(self, x: Tensor) -> Optional[Dict[str, Any]]:
        """
        Args:
            x: (L,F) anomalous window

        Returns:
            dict with x_cf, score, metadata
        """
        x = x.to(self.device)
        L, F = x.shape

        best = None

        for proto_idx, proto in enumerate(self.prototypes):
            for start in range(0, L - self.segment_length + 1):
                end = start + self.segment_length

                x_cf = x.clone()
                x_cf[start:end] = proto[start:end]

                score = self._score(x_cf)

                if score <= self.threshold:
                    # minimality = shortest segment first
                    return {
                        "x_cf": x_cf.detach().cpu(),
                        "score": score,
                        "edit": (start, end),
                        "meta": {
                            "method": "segment_substitution",
                            "prototype_index": proto_idx,
                            "segment": (start, end),
                        },
                    }
                    
                    
                  