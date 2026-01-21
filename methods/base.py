import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# =====================================================
# Utility functions
# =====================================================
def window_mse(x: Tensor, x_hat: Tensor) -> Tensor:
    """Reconstruction score per window (B,)"""
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
        """Compute reconstruction anomaly score."""
        x_hat = self.model(x)
        return window_mse(x, x_hat)

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
