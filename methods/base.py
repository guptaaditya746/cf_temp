import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# =====================================================
# Utility functions
# =====================================================
def window_mse(x: Tensor, x_hat: Tensor) -> Tensor:
    """Compute MSE per window. Works for both 2D and 3D tensors."""
    if x.ndim == 2:
        # Single window: (L, F)
        return ((x - x_hat) ** 2).mean()
    else:
        # Batch: (B, L, F)
        return ((x - x_hat) ** 2).mean(dim=(1, 2))


# =====================================================
# Base Counterfactual Class (ABSTRACT)
# =====================================================
class BaseCounterfactual(abc.ABC):
    """
    Gradient-free counterfactual base class for reconstruction models.

    Contract:
    - model: callable(x) -> x_hat
    - threshold: anomaly threshold τ
    """

    def __init__(
        self,
        model,
        threshold: float,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def score(self, x: Tensor) -> Tensor:
        """
        Compute reconstruction anomaly score.
        Handles both 2D (L, F) and 3D (B, L, F) inputs.

        Args:
            x: Input tensor, shape (L, F) or (B, L, F)

        Returns:
            Scalar tensor if input is 2D, (B,) tensor if input is 3D
        """
        # Check if input needs batch dimension
        needs_batch = x.ndim == 2

        if needs_batch:
            x = x.unsqueeze(0)  # (L, F) -> (1, L, F)

        # Ensure on correct device
        x = x.to(self.device)

        # Forward pass
        x_hat = self.model(x)

        # Compute MSE
        scores = window_mse(x, x_hat)

        # Return scalar for single window, tensor for batch
        if needs_batch:
            return scores.squeeze()  # Return scalar tensor
        return scores  # Return (B,) tensor

    def is_valid(self, x_cf: Tensor) -> bool:
        """Counterfactual validity check."""
        score = self.score(x_cf.unsqueeze(0))[0].item()
        return score <= self.threshold

    @abc.abstractmethod
    def generate(
        self,
        x: Tensor,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Must return:
        {
            "x_cf": Tensor (L,F),
            "score": float,
            "meta": dict
        }
        or None if no CF found.
        """
        raise NotImplementedError
