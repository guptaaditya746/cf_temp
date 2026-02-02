# methods/latent_cf.py
"""
Latent-space counterfactual wrapper for common CF API.
"""

from typing import Optional

import torch

from methods.latent.decoded_constraints import DecodedConstraintSpec
from methods.latent.latent_cf import RunnerConfig, generate_latent_counterfactual
from methods.latent.optimizers import CMAESConfig
from methods.latent.selection_and_validation import SelectionConfig


class LatentSpaceCounterfactual:
    """
    Latent-space counterfactual generator wrapped to match the common CF API.
    """

    def __init__(
        self,
        model,
        threshold,
        normal_windows,
        device,
        encoder,
        decoder,
        constraint_spec,
        cfg=None,
        score_fn=None,  # Optional: provide custom scoring function
    ):
        """
        Args:
            model: Anomaly detection model (for score_fn)
            threshold: Validity threshold (tau)
            normal_windows: Normal examples for reference (B, L, F)
            device: torch device
            encoder: Function x → z
            decoder: Function z → x_hat
            constraint_spec: DecodedConstraintSpec instance
            cfg: RunnerConfig (optional, defaults to scalar CMA-ES)
            score_fn: Optional custom scoring function (x -> float)
        """
        self.model = model
        self.threshold = float(threshold)
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.constraint_spec = constraint_spec
        self._custom_score_fn = score_fn

        # Default config if not provided
        if cfg is None:
            from methods.latent.latent_cf import RunnerConfig
            from methods.latent.optimizers import CMAESConfig
            from methods.latent.selection_and_validation import SelectionConfig

            cfg = RunnerConfig(
                mode="scalar_cmaes",
                eps_validity=0.05,
                latent_eps=1.5,
                seed=42,
                cmaes_cfg=CMAESConfig(
                    sigma0=0.3, max_evals=500, stop_on_first_valid=False
                ),
                selection_cfg=SelectionConfig(
                    require_validity=True,
                    robustness_trials=6,
                    robustness_sigma=0.03,
                    robustness_valid_frac=0.67,
                ),
            )
        self.cfg = cfg

        # Encode normal windows to latent space
        self.normalcore_z = self._encode_normal_core(normal_windows)

    def _encode_normal_core(self, normal_windows):
        """Encode normal windows to latent space (B, D)"""
        with torch.no_grad():
            # Ensure shape is (B, L, F)
            if normal_windows.ndim == 2:
                normal_windows = normal_windows.unsqueeze(0)

            z_list = []
            for i in range(normal_windows.shape[0]):
                z = self.encoder(normal_windows[i].to(self.device))
                z_list.append(z.cpu())

            return torch.stack(z_list, dim=0)

    def _score_fn(self, x):
        """Compute anomaly score for window x"""
        # ✅ USE CUSTOM SCORE FUNCTION IF PROVIDED
        if self._custom_score_fn is not None:
            return self._custom_score_fn(x)

        # Otherwise, try to auto-detect
        with torch.no_grad():
            if x.ndim == 2:
                x = x.unsqueeze(0)

            x_device = x.to(self.device)

            # Method 1: decision_function (sklearn-style)
            if hasattr(self.model, "decision_function"):
                score = self.model.decision_function(x_device.cpu().numpy())
                return float(score[0])

            # Method 2: Reconstruction-based (ReconstructionAnomalyModule)
            elif hasattr(self.model, "model"):
                reconstruction = self.model.model(x_device)
                mse = torch.mean((x_device - reconstruction) ** 2, dim=(1, 2))
                return float(mse[0].item())

            # Method 3: Direct forward pass
            else:
                reconstruction = self.model(x_device)
                mse = torch.mean((x_device - reconstruction) ** 2, dim=(1, 2))
                return float(mse[0].item())

    def generate(self, x_anomaly):
        """
        Generate counterfactual for x_anomaly.

        Args:
            x_anomaly: Tensor of shape (L, F)

        Returns:
            Result dict with keys: x_cf, z_cf, score, meta
            Returns None if generation fails
        """
        from methods.latent.latent_cf import generate_latent_counterfactual

        # Ensure input is on correct device
        x = x_anomaly.to(self.device)

        # Compute original score
        score_orig = self._score_fn(x)

        # Generate counterfactual
        result = generate_latent_counterfactual(
            x=x,
            encoder=self.encoder,
            decoder=self.decoder,
            score_fn=self._score_fn,
            tau=self.threshold,
            constraint_spec=self.constraint_spec,
            normalcore_z=self.normalcore_z,
            cfg=self.cfg,
        )

        # Add original score to meta
        result["meta"]["score_orig"] = score_orig

        # Add evaluation count if available
        if "n_evals" in result["meta"]:
            result["meta"]["evals"] = result["meta"]["n_evals"]

        # Return None if failed (to match other methods' API)
        if result["meta"]["status"] != "success":
            print(
                f"⚠️  CF generation failed: {result['meta'].get('failure_type', 'unknown')}"
            )
            print(
                f"⚠️  CF generation failed: {result['meta'].get('failure_type', 'unknown')}"
            )
            return None

        return result
