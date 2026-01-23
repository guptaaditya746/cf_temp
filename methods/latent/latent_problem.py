# latent_problem.py
# Part 1: Latent CF Problem Definition (implementation-ready)
#
# This module defines the *problem contract* for latent-space counterfactual search.
# It does NOT do optimization and does NOT enforce decoded constraints (that is Part 2).
#
# Core rules enforced here:
# - Valid CF is defined only by decoded validity: score(Decoder(z_cf)) <= tau - eps
# - Score is treated as black-box (no gradients assumed)
# - Latent edits are local (||z_cf - z|| <= latent_eps, if provided)
#
# External deps: torch, numpy (no need to reinvent basics)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

Tensor = torch.Tensor


def _as_2d_window(x: Tensor) -> Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"x must have shape (L, F). Got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("x contains NaN/Inf")
    return x


def _as_1d_latent(z: Tensor) -> Tensor:
    if not isinstance(z, torch.Tensor):
        raise TypeError("z must be a torch.Tensor")
    if z.ndim != 1:
        raise ValueError(f"z must have shape (D,). Got {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        raise ValueError("z contains NaN/Inf")
    return z


@dataclass(frozen=True)
class LatentSearchSpace:
    """
    Defines how we are allowed to edit latent code z.

    - bounds: optional per-dimension bounds for z_cf
      * shape: (D, 2) where [:,0]=low, [:,1]=high
    - groups: optional list of index arrays for grouped edits (interpretability)
    - editable_mask: optional boolean mask (D,) to freeze certain latent dims
    - latent_eps: optional locality radius; if set, enforce ||z_cf - z||_2 <= latent_eps in evaluation helpers
    """
    bounds: Optional[Tensor] = None

    bounds: Optional[Tensor] = None
    groups: Optional[Sequence[np.ndarray]] = None
    editable_mask: Optional[Tensor] = None
    latent_eps: Optional[float] = None

    def validate(
        self, latent_dim: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if self.bounds is not None:
            b = self.bounds
            if not isinstance(b, torch.Tensor):
                raise TypeError("bounds must be a torch.Tensor")
            if b.shape != (latent_dim, 2):
                raise ValueError(
                    f"bounds must have shape (D,2) = ({latent_dim},2). Got {tuple(b.shape)}"
                )
            if not torch.isfinite(b).all():
                raise ValueError("bounds contains NaN/Inf")
            low, high = b[:, 0], b[:, 1]
            if (low > high).any():
                raise ValueError("bounds invalid: some low > high")
        if self.editable_mask is not None:
            m = self.editable_mask
            if not isinstance(m, torch.Tensor):
                raise TypeError("editable_mask must be a torch.Tensor")
            if m.shape != (latent_dim,):
                raise ValueError(
                    f"editable_mask must have shape (D,) = ({latent_dim},). Got {tuple(m.shape)}"
                )
            if m.dtype != torch.bool:
                raise TypeError("editable_mask must be dtype=bool")
        if self.groups is not None:
            # groups are numpy arrays of indices (int)
            for g in self.groups:
                if not isinstance(g, np.ndarray):
                    raise TypeError("each group must be a numpy.ndarray of indices")
                if g.dtype.kind not in {"i", "u"}:
                    raise TypeError("group indices must be integer dtype")
                if g.min(initial=0) < 0 or g.max(initial=-1) >= latent_dim:
                    raise ValueError("group indices out of range")
        if self.latent_eps is not None and self.latent_eps <= 0:
            raise ValueError("latent_eps must be > 0 if provided")

        # Ensure tensors are on correct device/dtype in downstream usage (we don't mutate here).


@dataclass
class LatentCFProblem:
    """
    The canonical representation of the latent CF problem for one instance x.

    You pass this object to:
    - constraint evaluator (Part 2)
    - objective builder (Part 3)
    - optimizer (Part 4)
    """

    # Required callables
    encoder: Callable[[Tensor], Tensor]  # x(L,F) -> z(D,)
    decoder: Callable[[Tensor], Tensor]  # z(D,) -> x_hat(L,F)
    score_fn: Callable[
        [Tensor], float
    ]  # x(L,F) -> scalar score (higher = more anomalous)

    # Threshold rule
    tau: float  # decision threshold
    eps_validity: float = 1e-3  # margin: require score <= tau - eps_validity

    # Reference core (latent)
    normalcore_z: Optional[Tensor] = (
        None  # (K,D) tensor, optional but strongly recommended
    )

    # Latent editing constraints
    search_space: LatentSearchSpace = field(default_factory=LatentSearchSpace)

    # Diagnostics / bookkeeping
    meta: Dict[str, Any] = field(default_factory=dict)

    # Internal fields (set by build())
    x: Optional[Tensor] = None
    z: Optional[Tensor] = None
    latent_dim: Optional[int] = None

    def build(self, x: Tensor) -> "LatentCFProblem":
        """
        Bind the problem to a конкретный input window x, compute z=Encoder(x),
        validate shapes, and validate search space.
        """
        x = _as_2d_window(x)

        # Encode
        with torch.no_grad():
            z = self.encoder(x)
        z = _as_1d_latent(z)

        # Basic checks: decode roundtrip shape (optional but practical)
        with torch.no_grad():
            x_hat = self.decoder(z)
        x_hat = _as_2d_window(x_hat)
        if x_hat.shape != x.shape:
            raise ValueError(
                f"decoder(z) must have same shape as x: {tuple(x.shape)}. Got {tuple(x_hat.shape)}"
            )

        # Validate threshold parameters
        if not np.isfinite(self.tau):
            raise ValueError("tau must be finite")
        if not np.isfinite(self.eps_validity) or self.eps_validity <= 0:
            raise ValueError("eps_validity must be finite and > 0")
        if self.eps_validity >= 0.5 * abs(self.tau) and abs(self.tau) > 1e-9:
            # Not a hard law, but catches silly margins
            self.meta.setdefault("warnings", []).append(
                "eps_validity is large relative to tau; check scaling."
            )

        # Validate normalcore_z if provided
        if self.normalcore_z is not None:
            nz = self.normalcore_z
            if not isinstance(nz, torch.Tensor):
                raise TypeError("normalcore_z must be a torch.Tensor")
            if nz.ndim != 2:
                raise ValueError(
                    f"normalcore_z must have shape (K,D). Got {tuple(nz.shape)}"
                )
            if nz.shape[1] != z.shape[0]:
                raise ValueError(
                    f"normalcore_z D mismatch. Expected D={z.shape[0]} but got {nz.shape[1]}"
                )
            if not torch.isfinite(nz).all():
                raise ValueError("normalcore_z contains NaN/Inf")

        # Validate search space
        self.search_space.validate(
            latent_dim=z.shape[0], device=z.device, dtype=z.dtype
        )

        # Store bound instance
        self.x = x
        self.z = z
        self.latent_dim = z.shape[0]
        self.meta.setdefault("tau", float(self.tau))
        self.meta.setdefault("eps_validity", float(self.eps_validity))
        self.meta.setdefault("validity_target", float(self.tau - self.eps_validity))
        return self

    # -------------------------
    # Helpers used by later parts
    # -------------------------

    def validity_target(self) -> float:
        """Score must be <= this value to be considered valid."""
        return float(self.tau - self.eps_validity)

    def decode(self, z_cf: Tensor) -> Tensor:
        z_cf = _as_1d_latent(z_cf)
        if self.latent_dim is None:
            raise RuntimeError("Problem not built. Call build(x) first.")
        if z_cf.shape[0] != self.latent_dim:
            raise ValueError(
                f"z_cf must have shape (D,) with D={self.latent_dim}. Got {tuple(z_cf.shape)}"
            )
        with torch.no_grad():
            x_cf = self.decoder(z_cf)
        return _as_2d_window(x_cf)

    def score_decoded(self, z_cf: Tensor) -> float:
        """
        Black-box score evaluation of decoded candidate.
        No gradients assumed or used.
        """
        x_cf = self.decode(z_cf)
        s = float(self.score_fn(x_cf))
        if not np.isfinite(s):
            raise ValueError("score_fn returned non-finite value")
        return s

    def is_valid_cf(self, z_cf: Tensor) -> bool:
        """Decoded validity ONLY."""
        s = self.score_decoded(z_cf)
        return s <= self.validity_target()

    def apply_search_space(self, z_proposed: Tensor) -> Tensor:
        """
        Enforce simple latent-space constraints:
        - freeze dimensions via editable_mask
        - clamp to bounds
        This is not 'optimization'; it is projection.
        """
        if self.z is None:
            raise RuntimeError("Problem not built. Call build(x) first.")
        z0 = self.z
        z_p = _as_1d_latent(z_proposed).to(device=z0.device, dtype=z0.dtype)

        # Freeze non-editable dims
        if self.search_space.editable_mask is not None:
            m = self.search_space.editable_mask.to(device=z0.device)
            z_p = torch.where(m, z_p, z0)

        # Clamp bounds
        if self.search_space.bounds is not None:
            b = self.search_space.bounds.to(device=z0.device, dtype=z0.dtype)
            low, high = b[:, 0], b[:, 1]
            z_p = torch.max(torch.min(z_p, high), low)

        # Locality (optional) - projection to L2 ball around z0
        if self.search_space.latent_eps is not None:
            eps = float(self.search_space.latent_eps)
            d = z_p - z0
            norm = torch.linalg.norm(d)
            if norm > eps:
                z_p = z0 + d * (eps / (norm + 1e-12))

        return z_p

    def sample_initial_population(
        self,
        n: int,
        sigma: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> Tensor:
        """
        Generate n initial latent candidates around z, respecting search space.
        Intended for CMA-ES/NSGA-II initialization.

        Returns: (n, D) tensor
        """
        if self.z is None or self.latent_dim is None:
            raise RuntimeError("Problem not built. Call build(x) first.")
        if n <= 0:
            raise ValueError("n must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        rng = rng or np.random.default_rng()

        z0 = self.z
        D = self.latent_dim
        noise = torch.from_numpy(
            rng.normal(0.0, sigma, size=(n, D)).astype(np.float32)
        ).to(z0.device, z0.dtype)
        Z = z0.unsqueeze(0) + noise

        # Project each candidate to constraints
        Zp = torch.stack([self.apply_search_space(Z[i]) for i in range(n)], dim=0)
        return Zp

    # Optional: locality sanity check (decoder only)
    def sanity_check_decoder_locality(
        self,
        num_samples: int = 16,
        perturb_sigma: float = 0.1,
        decoded_distance_fn: Optional[Callable[[Tensor, Tensor], float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """
        Practical check: small z perturbations should not produce wild decoded changes.
        This does NOT prove validity, but catches obvious broken latent spaces early.

        decoded_distance_fn defaults to mean L2 per timestep-feature.
        """
        if self.z is None or self.x is None:
            raise RuntimeError("Problem not built. Call build(x) first.")
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if perturb_sigma <= 0:
            raise ValueError("perturb_sigma must be > 0")
        rng = rng or np.random.default_rng()

        if decoded_distance_fn is None:

            def decoded_distance_fn(a: Tensor, b: Tensor) -> float:
                return float(torch.mean((a - b) ** 2).sqrt().item())

        z0 = self.z
        x0 = self.decode(z0)

        D = z0.shape[0]
        noise = torch.from_numpy(
            rng.normal(0.0, perturb_sigma, size=(num_samples, D)).astype(np.float32)
        ).to(z0.device, z0.dtype)
        dists = []
        for i in range(num_samples):
            z_i = self.apply_search_space(z0 + noise[i])
            x_i = self.decode(z_i)
            dists.append(decoded_distance_fn(x0, x_i))

        dists_np = np.asarray(dists, dtype=np.float64)
        out = {
            "decoded_dist_mean": float(dists_np.mean()),
            "decoded_dist_std": float(dists_np.std(ddof=1) if num_samples > 1 else 0.0),
            "decoded_dist_max": float(dists_np.max()),
            "num_samples": float(num_samples),
            "perturb_sigma": float(perturb_sigma),
        }
        return out


# -------------------------
# Convenience constructor
# -------------------------


def define_latent_cf_problem(
    *,
    encoder: Callable[[Tensor], Tensor],
    decoder: Callable[[Tensor], Tensor],
    score_fn: Callable[[Tensor], float],
    tau: float,
    eps_validity: float = 1e-3,
    normalcore_z: Optional[Tensor] = None,
    bounds: Optional[Union[np.ndarray, Tensor]] = None,
    editable_mask: Optional[Union[np.ndarray, Tensor]] = None,
    groups: Optional[Sequence[np.ndarray]] = None,
    latent_eps: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> LatentCFProblem:
    """
    Build the unbound problem object (bind to x via problem.build(x)).
    """
    b = None
    if bounds is not None:
        b = (
            bounds
            if isinstance(bounds, torch.Tensor)
            else torch.from_numpy(np.asarray(bounds))
        )
    m = None
    if editable_mask is not None:
        if isinstance(editable_mask, torch.Tensor):
            m = editable_mask
        else:
            m = torch.from_numpy(np.asarray(editable_mask).astype(bool))
    ss = LatentSearchSpace(
        bounds=b, groups=groups, editable_mask=m, latent_eps=latent_eps
    )

    return LatentCFProblem(
        encoder=encoder,
        decoder=decoder,
        score_fn=score_fn,
        tau=float(tau),
        eps_validity=float(eps_validity),
        normalcore_z=normalcore_z,
        search_space=ss,
        meta=meta or {},
    )


# -------------------------
# Minimal usage example (remove in production)
# -------------------------
if __name__ == "__main__":
    # Dummy encoder/decoder/score for smoke test
    L, F, D = 32, 3, 8
    torch.manual_seed(0)

    class DummyAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(L * F, D)
            )
            self.dec = torch.nn.Sequential(torch.nn.Linear(D, L * F))

        def encode(self, x):
            return self.enc(x)

        def decode(self, z):
            x = self.dec(z).view(L, F)
            return x

    ae = DummyAE()

    def encoder(x: Tensor) -> Tensor:
        return ae.encode(x)

    def decoder(z: Tensor) -> Tensor:
        return ae.decode(z)

    def score_fn(x: Tensor) -> float:
        # toy "anomaly score" = L2 magnitude
        return float(torch.mean(x**2).sqrt().item())

    x = torch.randn(L, F)
    normalcore_x = torch.randn(64, L, F)
    with torch.no_grad():
        normalcore_z = torch.stack(
            [encoder(normalcore_x[i]) for i in range(normalcore_x.shape[0])], dim=0
        )
    problem = define_latent_cf_problem(
        encoder=encoder,
        decoder=decoder,
        score_fn=score_fn,
        tau=1.0,
        eps_validity=0.05,
        normalcore_z=normalcore_z,
        latent_eps=1.5,
        meta={"name": "dummy_problem"},
    ).build(x)

    print("z dim:", problem.latent_dim)
    print("validity target:", problem.validity_target())
    print("decoder locality check:", problem.sanity_check_decoder_locality())
    Z0 = problem.sample_initial_population(5, sigma=0.2)
    print("init pop shape:", tuple(Z0.shape))
