
import torch
from torch import Tensor
from typing import Optional, Dict, Any


class NearestPrototypeCounterfactual:
    """
    Instance-based, gradient-free counterfactual.

    Strategy:
    - Iterate normal prototypes by increasing distance
    - Return the nearest one that satisfies score <= threshold
    """

    def __init__(
        self,
        model,
        threshold: float,
        normal_windows: Tensor,  # (K, L, F)
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.device = device or torch.device("cpu")

        assert normal_windows.ndim == 3, "normal_windows must be (K,L,F)"
        self.prototypes = normal_windows.to(self.device)

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

        # Distance to prototypes (for minimality)
        distances = ((self.prototypes - x.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        sorted_idx = torch.argsort(distances)

        for idx in sorted_idx:
            candidate = self.prototypes[idx]
            score = self._score(candidate)

            if score <= self.threshold:
                return {
                    "x_cf": candidate.detach().cpu(),
                    "score": score,
                    "distance": float(distances[idx].item()),
                    "meta": {
                        "method": "nearest_prototype",
                        "prototype_index": int(idx),
                    },
                }

        return None
      
