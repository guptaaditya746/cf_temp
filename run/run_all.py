# ============================================================================
# IMPORTS
# ============================================================================

# --- Standard Library ---
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightning.pytorch as pl

# --- Third-Party: Scientific Computing ---
import numpy as np
import pandas as pd

# --- Third-Party: PyTorch & Lightning ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from methods.comte.Candidate_Selection_Validation import (
    CandidateSelector as CoMTESelector,
)
from methods.comte.Candidate_Selection_Validation import (
    SelectionConfig as CoMTESelectionConfig,
)
from methods.comte.Constraint_Evaluation import (
    ConstraintConfig as CoMTEConstraintConfig,
)
from methods.comte.Constraint_Evaluation import (
    ConstraintEvaluator as CoMTEConstraintEvaluator,
)
from methods.comte.Failure_Handling import FailureHandler
from methods.comte.NormalCore_Donor_Matching import (
    DonorMatchConfig,
    NormalCoreDonorMatcher,
)
from methods.comte.Segment_Candidate_Generation import (
    SegmentCandidateGenerator,
    SegmentGenConfig,
)
from methods.comte.Segment_Substitution_Engine import (
    SegmentSubstitutor,
    SubstitutionConfig,
)
from methods.comte_final import CoMTEReconstructionCF
from methods.generative.candidate_selection import (
    CandidateSelector as GenerativeSelector,
)
from methods.generative.candidate_selection import (
    SelectionConfig as GenerativeSelectionConfig,
)
from methods.generative.constraint_evaluation import (
    ConstraintConfig,
    ConstraintEvaluator,
)

# --- Local: Utilities ---
from utils.metrics import CounterfactualMetrics, MetricsConfig
from utils.plot_counterfactual import plot_counterfactual

# --- Warnings Configuration ---
# Suppress specific warnings only, not all
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
# Keep FutureWarning, DeprecationWarning visible for debugging
# Add these missing imports:
from methods.generative.candidate_selection import (
    CandidateSelector as GenerativeSelector,
)
from methods.generative.candidate_selection import (
    SelectionConfig as GenerativeSelectionConfig,
)
from methods.generative.constraint_evaluation import (
    ConstraintConfig,
    ConstraintEvaluator,
)
from methods.generative.infilling_engine import InfillingConfig, InfillingEngine
from methods.generative.mask_strategy import MaskStrategy, MaskStrategyConfig
from methods.generative_final import GenerativeInfillingCounterfactual
from methods.genetic.optimizer_nsga2 import NSGA2Config, NSGA2Optimizer
from methods.genetic.sensor_constraints import SensorConstraintManager
from methods.genetic_cf import SegmentCounterfactualProblem
from methods.latent.decoded_constraints import DecodedConstraintSpec
from methods.latent.latent_cf import RunnerConfig
from methods.latent.optimizers import CMAESConfig
from methods.latent_final import LatentSpaceCounterfactual
from methods.motif import MotifGuidedSegmentRepairCF
from methods.nearest_prototype import NearestPrototypeCounterfactual
from methods.segment_substitution import SegmentSubstitutionCounterfactual

# ============================================================================
# CONFIGURATION
# ============================================================================
# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# DATAFILE = os.path.join(
#     DATA_DIR,
#     "ATACAMA",
#     "Atacama.pkl"
# )
DATAFILE = (
    "/home/gupt_ad/conclusion_work/application/ad-xai_pipeline/ATACAMA/Atacama.pkl"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "benchmark_all_methods")

# --- Data Processing ---
WINDOW_SIZE = 64
STEP_SIZE = 10
FEATURE_COLUMNS = [
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "precipitation (mm)",
    "pressure_msl (hPa)",
    "cloud_cover (%)",
    "wind_speed_10m (km/h)",
]
# --- Benchmark Parameters ---
N_NORMAL_CORE = 200  # Size of reference normal dataset
N_TEST_ANOMALIES = 1  # Number of test anomalies per method
RANDOM_SEED = 42


# --- Model Architecture ---
@dataclass
class AnomalyConfig:
    """LSTM Autoencoder configuration."""

    # Data
    batch_size: int = 512
    num_workers: int = 16
    input_size: int = 6  # Number of features

    # Model Architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    # Training
    max_epochs: int = 2  # Quick training for benchmark
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = RANDOM_SEED

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.lr > 0, "learning rate must be positive"
        assert self.num_layers > 0, "num_layers must be positive"

        # Warn about potential issues
        if self.max_epochs == 1:
            warnings.warn("max_epochs=1 is very low. Model may underfit.", UserWarning)


# --- Device Configuration ---
def get_device() -> torch.device:
    """Get optimal device with fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================


class DataValidator:
    """Validates data quality and shapes."""

    @staticmethod
    def check_windows(X: np.ndarray, expected_window_size: int) -> None:
        """Validate window array shape and values."""
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N, L, F), got shape {X.shape}")

        if X.shape[1] != expected_window_size:
            raise ValueError(
                f"Window size mismatch: expected {expected_window_size}, "
                f"got {X.shape[1]}"
            )

        if not np.isfinite(X).all():
            n_nan = np.isnan(X).sum()
            n_inf = np.isinf(X).sum()
            raise ValueError(f"Data contains {n_nan} NaN and {n_inf} Inf values")

    @staticmethod
    def check_labels(y: np.ndarray, X: np.ndarray) -> None:
        """Validate label array."""
        if len(y) != len(X):
            raise ValueError(
                f"Label count ({len(y)}) doesn't match data count ({len(X)})"
            )

        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"Labels must be binary (0/1), got {unique_labels}")


def load_and_preprocess(
    filepath: str, window_size: int, step_size: int, cache: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads Atacama data with validation and optional caching.

    Args:
        filepath: Path to pickle file
        window_size: Sliding window length
        step_size: Window stride
        cache: If True, save processed data to .npz for faster reloading

    Returns:
        X: (N, L, F) windowed features
        y: (N, L) windowed labels

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data validation fails
    """
    cache_path = filepath.replace(".pkl", f"_w{window_size}_s{step_size}.npz")

    # Try loading from cache first
    if cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path)
        return data["X"], data["y"]

    print(f"Loading data from {filepath}...")

    # Load raw data
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_pickle(filepath)

    # Binarize labels
    label_mapping = {0: 0, 1: 1, 2: 1}
    df["binary_label"] = df["label"].map(label_mapping).fillna(0).astype(int)

    # Extract features
    available_features = [c for c in df.columns if any(f in c for f in FEATURE_COLUMNS)]

    if not available_features:
        raise ValueError(
            f"No features found matching {FEATURE_COLUMNS}. "
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Using features: {available_features}")

    X_raw = df[available_features].values
    y_raw = df["binary_label"].values

    # Check for NaN/Inf before standardization
    if not np.isfinite(X_raw).all():
        print(f"Warning: Filling {np.isnan(X_raw).sum()} NaN values with column mean")
        X_raw = pd.DataFrame(X_raw).fillna(method="ffill").fillna(method="bfill").values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Sliding window
    X_windows_raw = sliding_window_view(X_scaled, window_shape=window_size, axis=0)[
        ::step_size
    ]
    X_windows = np.moveaxis(X_windows_raw, -2, -1)  # (N, L, F)

    y_windows = sliding_window_view(y_raw, window_shape=window_size, axis=0)[
        ::step_size
    ]
    X_scaled = scaler.fit_transform(X_raw)

    # Sliding window
    X_windows_raw = sliding_window_view(X_scaled, window_shape=window_size, axis=0)[
        ::step_size
    ]
    X_windows = np.moveaxis(X_windows_raw, -2, -1)  # (N, L, F)

    y_windows = sliding_window_view(y_raw, window_shape=window_size, axis=0)[
        ::step_size
    ]

    # Validate
    DataValidator.check_windows(X_windows, window_size)
    DataValidator.check_labels(y_windows, X_windows)

    print(f"✓ X shape: {X_windows.shape}")
    print(f"✓ y shape: {y_windows.shape}")

    # Cache for next time
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X_windows, y=y_windows)
        print(f"✓ Cached to {cache_path}")

    return X_windows, y_windows


def prepare_labels(y_windows: np.ndarray) -> np.ndarray:
    """
    Label window as anomaly if ANY timestep is anomalous.

    Args:
        y_windows: (N, L) or (N,) labels

    Returns:
        (N,) binary labels
    """
    if y_windows.ndim == 2:
        return (y_windows.sum(axis=1) > 0).astype(np.int64)
    elif y_windows.ndim == 1:
        return y_windows.astype(np.int64)
    else:
        raise ValueError(f"Expected 1D or 2D labels, got shape {y_windows.shape}")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================


class LSTMAutoencoder(nn.Module):
    """LSTM-based sequence autoencoder for anomaly detection."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()

        # Store config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projection layer
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, F) input sequence

        Returns:
            x_hat: (B, L, F) reconstructed sequence
        """
        # Validate input
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, L, F), got shape {x.shape}")

        # Encode
        _, (h_n, c_n) = self.encoder(x)
        z = h_n[-1]  # (B, H) - use last layer's hidden state

        # Decode
        B, L, _ = x.shape
        dec_in = z.unsqueeze(1).repeat(1, L, 1)  # (B, L, H)
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))

        # Project back to input space
        x_hat = self.proj(dec_out)

        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation."""
        with torch.no_grad():
            _, (h_n, _) = self.encoder(x)
            return h_n[-1]  # (B, H)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode from latent space."""
        with torch.no_grad():
            B = z.shape[0]
            dec_in = z.unsqueeze(1).repeat(1, seq_len, 1)
            h_n = z.unsqueeze(0).repeat(self.num_layers, 1, 1)
            c_n = torch.zeros_like(h_n)
            dec_out, _ = self.decoder(dec_in, (h_n, c_n))
            return self.proj(dec_out)


class WindowDataset(Dataset):
    """PyTorch dataset for windowed time series."""

    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None):
        self.x = torch.tensor(windows, dtype=torch.float32)
        self.y = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx], self.x[idx]  # Return x twice for autoencoder


class ReconstructionAnomalyModule(pl.LightningModule):
    """Lightning module for LSTM autoencoder training."""

    def __init__(self, cfg: AnomalyConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Model
        self.model = LSTMAutoencoder(
            cfg.input_size, cfg.hidden_size, cfg.num_layers, cfg.dropout
        )

        # ⚠️ CRITICAL FIX: Remove self.device assignment
        # Lightning manages device automatically via self.device property

        # Buffers for threshold tracking
        self.register_buffer("optimal_thresh", torch.tensor(0.0))
        self.val_scores = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        scores = ((x - x_hat) ** 2).mean(dim=(1, 2))
        self.val_scores.append(scores.cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if self.val_scores:
            all_scores = torch.cat(self.val_scores)
            thresh = torch.quantile(all_scores, 0.95)
            self.optimal_thresh.fill_(thresh)
            self.val_scores.clear()
            self.log("val_optimal_thresh", thresh)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> float:
        """Compute reconstruction anomaly score (MSE) for a single window."""
        self.eval()
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # ✅ USE LIGHTNING'S DEVICE PROPERTY (not self.device from __init__)
        x = x.to(self.device)
        x_hat = self(x)
        score_val = ((x - x_hat) ** 2).mean()
        return float(score_val.item())

    # In your ReconstructionAnomalyModule class
    def get_performance_summary(self):
        """Returns a dictionary of key training metrics."""
        return {
            "best_val_f1": float(self.best_val_f1.item())
            if hasattr(self, "best_val_f1")
            else None,
            "optimal_threshold": float(self.optimal_thresh.item()),
            "input_size": self.cfg.input_size,
            "hidden_size": self.cfg.hidden_size,
        }

    def export_torchscript(self, file_path: str):
        """Exports the underlying model to TorchScript format."""
        self.eval()
        # Create example input for tracing
        example_input = torch.randn(1, 512, self.cfg.input_size).to(self.device)
        scripted_model = torch.jit.trace(self.model, example_input)
        scripted_model.save(file_path)
        print(f"✓ Model and weights saved in TorchScript: {file_path}")


class GenerativeAdapter:
    def __init__(self, pipeline, model, threshold, device):
        self.pipeline = pipeline
        self.model = model
        self.threshold = threshold
        self.device = device

    def anomaly_scorer(self, x_np: np.ndarray) -> float:
        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x_hat = self.model(x_t)
            return F.mse_loss(x_hat, x_t).item()

    def reconstruction_error(self, x_np: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x_hat = self.model(x_t)
            err_t = (x_t - x_hat).pow(2).mean(dim=-1).squeeze(0)  # shape (L,)
        return err_t.cpu().numpy()

    def generate(self, x_t: torch.Tensor):
        x_np = x_t.detach().cpu().numpy()  # (L, F)
        et = self.reconstruction_error(x_np)  # (L,)
        cf_res = self.pipeline.generate(
            x=x_np,
            reconstruction_error_t=et,
            anomaly_score_fn=self.anomaly_scorer,
            threshold=float(self.threshold),
        )
        if cf_res is None:
            return None
        return {
            "xcf": torch.tensor(cf_res.x_cf, dtype=torch.float32),
            "score": cf_res.score,
            "meta": cf_res.meta,
        }


# ============================================================================
# TRAINING PIPELINE
# ============================================================================


def train_model(
    cfg: AnomalyConfig,
    windows: np.ndarray,
    val_split: float = 0.2,
    verbose: bool = True,
) -> ReconstructionAnomalyModule:
    """
    Train LSTM autoencoder on normal windows with validation.

    Args:
        cfg: Training configuration
        windows: Normal windows (N, L, F)
        val_split: Validation split fraction
        verbose: Whether to print progress

    Returns:
        Trained Lightning module

    Raises:
        RuntimeError: If training fails
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training LSTM Autoencoder")
        print(f"{'=' * 60}")
        print(f"  Training samples: {len(windows)}")
        print(f"  Window size: {windows.shape[1]}")
        print(f"  Features: {windows.shape[2]}")
        print(f"  Validation split: {val_split:.1%}")

    # Create datasets
    dataset = WindowDataset(windows)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),  # Reproducible splits
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,  # Faster multi-epoch training
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    # Initialize model
    model = ReconstructionAnomalyModule(cfg)

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,  # Disable logging for benchmark
        enable_checkpointing=False,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        deterministic=True,  # Reproducibility
        gradient_clip_val=1.0,  # Prevent exploding gradients
    )

    # Train with error handling
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}") from e
    finally:
        # Clean up dataloaders to free memory
        del train_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if verbose:
        print(f"  ✓ Training complete")
        print(f"  ✓ Optimal threshold: {model.optimal_thresh.item():.4f}")

    return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def safe_float(x: Any) -> Optional[float]:
    """
    Convert various numeric types to float with robust error handling.

    Args:
        x: Value to convert (int, float, np.number, torch.Tensor, etc.)

    Returns:
        Float value or None if conversion fails
    """
    if x is None:
        return None

    # Direct numeric types
    if isinstance(x, (float, int)):
        return float(x)

    # NumPy types
    if isinstance(x, (np.floating, np.integer)):
        return float(x)

    # PyTorch tensors
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.item())
        else:
            # Multi-element tensor - take mean
            warnings.warn(f"Converting multi-element tensor to float (taking mean)")
            return float(x.mean().item())

    # Boolean to float
    if isinstance(x, bool):
        return 1.0 if x else 0.0

    # Try generic conversion
    try:
        return float(x)
    except (ValueError, TypeError):
        warnings.warn(f"Could not convert {type(x).__name__} to float")
        return None


def collect_metric_values(metrics_list: List[Dict[str, Any]], key: str) -> List[float]:
    """
    Extract specific metric from list of metric dicts.

    Args:
        metrics_list: List of metric dictionaries
        key: Metric key to extract

    Returns:
        List of valid float values (skips None)
    """
    values = []
    for m in metrics_list:
        if m is None:
            continue

        val = m.get(key)
        float_val = safe_float(val)

        if float_val is not None and not np.isnan(float_val):
            values.append(float_val)

    return values


def compute_statistics(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Compute robust statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with mean, median, min, max, std
    """
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "count": 0,
        }

    arr = np.array(values, dtype=np.float64)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "count": len(arr),
    }


def summarize_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate list of metric dicts into summary statistics.

    Args:
        metrics_list: List of metric dictionaries from different runs

    Returns:
        Dictionary with aggregated statistics for each metric
    """
    if not metrics_list:
        return {"n": 0, "found_rate": 0.0, "valid_rate": 0.0}

    # Filter out None entries
    valid_metrics = [m for m in metrics_list if m is not None]
    n_total = len(metrics_list)
    n_valid = len(valid_metrics)

    summary = {
        "n": n_total,
        "n_valid": n_valid,
        "found_rate": n_valid / n_total if n_total > 0 else 0.0,
    }

    # Metrics to aggregate
    metric_keys = [
        "valid",
        "score_cf",
        "delta_score_to_thr",
        "dist_rmse",
        "dist_mae",
        "frac_changed",
        "n_segments",
        "smooth_l2_d1",
        "nn_dist_to_normal_core",
        "z_abs_mean",
        "evals",
    ]

    # Compute statistics for each metric
    for key in metric_keys:
        values = collect_metric_values(valid_metrics, key)

        if key == "valid":
            # Boolean metric - compute rate
            summary["valid_rate"] = np.mean(values) if values else 0.0
        else:
            # Numeric metric - compute full statistics
            summary[key] = compute_statistics(values)

    return summary


def format_metric_value(value: Optional[float], format_type: str = "float") -> str:
    """
    Format metric value for display with appropriate precision.

    Args:
        value: Numeric value to format
        format_type: One of "float", "percent", "int"

    Returns:
        Formatted string
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"

    if format_type == "float":
        return f"{value:.4f}"
    elif format_type == "percent":
        return f"{value * 100:.1f}%"
    elif format_type == "int":
        return f"{int(value)}"
    else:
        return str(value)


# ============================================================================
# METHOD INITIALIZATION
# ============================================================================

# ============================================================================
# METHOD INITIALIZATION
# ============================================================================


@dataclass
class MethodConfig:
    """Configuration for individual CF methods."""

    segment_length: int = 8
    max_segments_per_len: int = 12
    top_motifs_per_segment: int = 8
    motif_lengths: List[int] = (4, 6, 8, 10, 12, 16, 20, 24)

    # Genetic algorithm
    genetic_pop_size: int = 50
    genetic_n_gen: int = 30
    genetic_crossover_prob: float = 0.9

    # CMA-ES
    cmaes_max_evals: int = 500
    latent_eps: float = 0.5

    # Generative
    n_infill_samples: int = 12
    mask_token_mode: str = "zeros"

    # CoMTE
    comte_max_donors: int = 5
    comte_smoothing_alpha: float = 0.4


def create_encoder_decoder_wrappers(
    model: ReconstructionAnomalyModule,
    window_size: int,
    device: torch.device,
    return_device: str = "cpu",  # ✅ Always return on CPU for optimizer
) -> Tuple[Callable, Callable]:
    """
    Create encoder/decoder wrappers for latent-space methods.

    ⚠️ CRITICAL: Wrappers return tensors on CPU to avoid device mismatch
    with optimization libraries that expect CPU tensors.

    Args:
        model: Trained autoencoder
        window_size: Sequence length
        device: Device where model is located
        return_device: Device for output tensors (default: "cpu")

    Returns:
        (encoder_fn, decoder_fn) tuple
    """
    return_dev = torch.device(return_device)

    def encoder_wrapper(x: torch.Tensor) -> torch.Tensor:
        """Encode (L, F) → (D,) latent vector."""
        with torch.no_grad():
            if x.ndim == 2:
                x = x.unsqueeze(0)  # Add batch dim

            x = x.to(device)
            _, (h_n, _) = model.model.encoder(x)
            z = h_n[-1].squeeze(0)  # Remove batch dim

            # ✅ CRITICAL FIX: Move to CPU before returning
            return z.to(return_dev)

    def decoder_wrapper(z: torch.Tensor) -> torch.Tensor:
        """Decode (D,) latent → (L, F) sequence."""
        with torch.no_grad():
            if z.ndim == 1:
                z = z.unsqueeze(0)  # Add batch dim

            z = z.to(device)
            L = window_size

            # Prepare decoder input
            dec_in = z.unsqueeze(1).repeat(1, L, 1)
            h_n = z.unsqueeze(0).repeat(model.cfg.num_layers, 1, 1)
            c_n = torch.zeros_like(h_n).to(device)

            # Decode
            dec_out, _ = model.model.decoder(dec_in, (h_n, c_n))
            x_hat = model.model.proj(dec_out).squeeze(0)  # Remove batch dim

            # ✅ CRITICAL FIX: Move to CPU before returning
            return x_hat.to(return_dev)

    return encoder_wrapper, decoder_wrapper


def initialize_all_methods(
    model: ReconstructionAnomalyModule,
    threshold: float,
    normal_windows: torch.Tensor,
    device: torch.device,
    X_train: np.ndarray,
    cfg: Optional[MethodConfig] = None,
) -> Dict[str, Any]:
    """
    Initialize all 7 counterfactual methods with proper configuration.

    Args:
        model: Trained anomaly detection model
        threshold: Anomaly score threshold
        normal_windows: Reference normal windows (K, L, F) on device
        device: PyTorch device
        X_train: Training data for constraint calculation (N, L, F)
        cfg: Method-specific configuration

    Returns:
        Dictionary mapping method names to initialized instances
    """
    if cfg is None:
        cfg = MethodConfig()

    print(f"\n{'=' * 60}")
    print("Initializing Counterfactual Methods")
    print(f"{'=' * 60}")

    all_methods = {}

    # -------------------------------------------------------------------------
    # 1. Segment Substitution
    # -------------------------------------------------------------------------
    try:
        all_methods["Segment Substitution"] = SegmentSubstitutionCounterfactual(
            model=model,
            threshold=threshold,
            normal_windows=normal_windows,
            segment_length=cfg.segment_length,
            device=device,
        )
        print("  ✓ Segment Substitution")
    except Exception as e:
        print(f"  ✗ Segment Substitution failed: {e}")

    # -------------------------------------------------------------------------
    # 2. Nearest Prototype
    # -------------------------------------------------------------------------
    try:
        all_methods["Nearest Prototype"] = NearestPrototypeCounterfactual(
            model=model,
            threshold=threshold,
            normal_windows=normal_windows,
            device=device,
        )
        print("  ✓ Nearest Prototype")
    except Exception as e:
        print(f"  ✗ Nearest Prototype failed: {e}")

    # -------------------------------------------------------------------------
    # 3. Motif-Guided Segment Repair
    # -------------------------------------------------------------------------
    try:
        all_methods["Motif-Guided"] = MotifGuidedSegmentRepairCF(
            model=model,
            threshold=threshold,
            device=device,
            normal_core=normal_windows,
            max_segments_per_len=cfg.max_segments_per_len,
            top_motifs_per_segment=cfg.top_motifs_per_segment,
            lengths=cfg.motif_lengths,
            edge_blend=2,
            use_error_guidance=True,
        )
        print("  ✓ Motif-Guided")
    except Exception as e:
        print(f"  ✗ Motif-Guided failed: {e}")

    # -------------------------------------------------------------------------
    # 4. Genetic (NSGA-II)
    # -------------------------------------------------------------------------
    try:
        optimizer = NSGA2Optimizer(
            config=NSGA2Config(
                pop_size=cfg.genetic_pop_size,
                n_gen=cfg.genetic_n_gen,
                crossover_prob=cfg.genetic_crossover_prob,
            )
        )
        constraints = SensorConstraintManager(normal_core=normal_windows, device=device)

        all_methods["Genetic (NSGA-II)"] = SegmentCounterfactualProblem(
            model=model,
            threshold=threshold,
            normal_core=normal_windows,
            constraints=constraints,
            optimizer=optimizer,
            device=str(device),
            eps_valid=0.5,
        )
        print("  ✓ Genetic (NSGA-II)")
    except Exception as e:
        print(f"  ✗ Genetic (NSGA-II) failed: {e}")

    # -------------------------------------------------------------------------
    # 5. Latent-Space (CMA-ES)
    # -------------------------------------------------------------------------
    try:
        # ✅ Create wrappers that return CPU tensors
        encoder_wrapper, decoder_wrapper = create_encoder_decoder_wrappers(
            model=model,
            window_size=WINDOW_SIZE,
            device=device,
            return_device="cpu",  # Critical fix
        )

        # Calculate feature bounds
        mins = torch.tensor(X_train.min(axis=(0, 1)), dtype=torch.float32)
        maxs = torch.tensor(X_train.max(axis=(0, 1)), dtype=torch.float32)
        roc = torch.tensor(
            np.percentile(np.diff(X_train, axis=1), 99, axis=0).max(axis=0),
            dtype=torch.float32,
        )

        constraint = DecodedConstraintSpec(
            value_min=mins - 0.5, value_max=maxs + 0.5, max_rate_of_change=roc
        )

        all_methods["Latent-Space (CMA-ES)"] = LatentSpaceCounterfactual(
            model=model,
            threshold=threshold,
            normal_windows=normal_windows[:200].cpu(),  # Move to CPU
            encoder=encoder_wrapper,
            decoder=decoder_wrapper,
            constraint_spec=constraint,
            cfg=RunnerConfig(
                mode="scalar_cmaes",
                latent_eps=cfg.latent_eps,
                cmaes_cfg=CMAESConfig(max_evals=cfg.cmaes_max_evals),
            ),
            device=device,
        )
        print("  ✓ Latent-Space (CMA-ES)")
    except Exception as e:
        print(f"  ✗ Latent-Space (CMA-ES) failed: {e}")
        import traceback

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # 6. Generative Infilling
    # -------------------------------------------------------------------------
    try:
        all_methods["Generative Infilling"] = initialize_generative_method(
            model=model,
            threshold=threshold,
            normal_windows=normal_windows,
            device=device,
            X_train=X_train,
            cfg=cfg,
        )
        print("  ✓ Generative Infilling")
    except Exception as e:
        print(f"  ✗ Generative Infilling failed: {e}")
        import traceback

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # 7. CoMTE
    # -------------------------------------------------------------------------
    try:
        all_methods["CoMTE"] = initialize_comte_method(
            model=model,
            threshold=threshold,
            normal_windows=normal_windows,
            device=device,
            X_train=X_train,
            cfg=cfg,
        )
        print("  ✓ CoMTE")
    except Exception as e:
        print(f"  ✗ CoMTE failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n✓ Initialized {len(all_methods)}/{7} methods")

    if len(all_methods) < 7:
        warnings.warn(
            f"Only {len(all_methods)}/7 methods initialized successfully", UserWarning
        )

    return all_methods


def initialize_generative_method(
    model: ReconstructionAnomalyModule,
    threshold: float,
    normal_windows: torch.Tensor,
    device: torch.device,
    X_train: np.ndarray,
    cfg: MethodConfig,
) -> GenerativeAdapter:
    """Initialize Generative Infilling method (separate function for clarity)."""

    # A. Mask Strategy
    mask_strategy = MaskStrategy(MaskStrategyConfig(random_seed=42))

    # B. Infilling Engine
    feature_means = normal_windows.detach().cpu().mean(dim=(0, 1)).numpy()

    class InfillerWrapper(nn.Module):
        """Wraps LSTM autoencoder to accept (x, mask) signature."""

        def __init__(self, autoencoder):
            super().__init__()
            self.autoencoder = autoencoder

        def forward(self, x, mask=None):
            return self.autoencoder(x)

    infilling_engine = InfillingEngine(
        infiller=InfillerWrapper(model),
        cfg=InfillingConfig(
            device=str(device),
            dtype=torch.float32,
            mask_token_mode=cfg.mask_token_mode,
            n_samples=cfg.n_infill_samples,
            sampling_seed=42,
            deterministic=False,
            smooth_masked=True,
            smooth_sigma=1.0,
            clamp_output=False,
        ),
        normalcore_feature_mean=feature_means,
    )

    # C. Constraint Evaluator
    constraint_eval = ConstraintEvaluator(
        cfg=ConstraintConfig(
            value_min=float(X_train.min()),
            value_max=float(X_train.max()),
            immutable_features=None,
        )
    )

    # D. Candidate Selector
    selector = GenerativeSelector(
        GenerativeSelectionConfig(
            mode="lexicographic",
            threshold=0,
            w_score_excess=10.0,
            w_mask_size=1.0,
            w_soft_penalty=1.0,
        ),
        score_fn=None,
    )

    # E. Assemble pipeline
    pipeline = GenerativeInfillingCounterfactual(
        mask_strategy=mask_strategy,
        infilling_engine=infilling_engine,
        constraint_evaluator=constraint_eval,
        candidate_selector=selector,
        failure_handler=FailureHandler(),
    )

    return GenerativeAdapter(
        pipeline=pipeline, model=model, threshold=threshold, device=device
    )


def initialize_comte_method(
    model: ReconstructionAnomalyModule,
    threshold: float,
    normal_windows: torch.Tensor,
    device: torch.device,
    X_train: np.ndarray,
    cfg: MethodConfig,
) -> CoMTEReconstructionCF:
    """Initialize CoMTE method (separate function for clarity)."""

    def reconstructor(x_input) -> torch.Tensor:
        if isinstance(x_input, np.ndarray):
            x_t = torch.tensor(x_input, dtype=torch.float32)
        else:
            x_t = x_input

        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)

        x_t = x_t.to(device)
        with torch.no_grad():
            x_hat = model(x_t)
        return x_hat.squeeze(0)

    def reconstruction_score(x_input) -> float:
        if isinstance(x_input, np.ndarray):
            x_t = torch.tensor(x_input, dtype=torch.float32)
        else:
            x_t = x_input

        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)

        x_t = x_t.to(device)
        with torch.no_grad():
            x_hat = model(x_t)
        return F.mse_loss(x_hat, x_t).item()

    # Part 1: Segment Generator
    base_seg_gen = SegmentCandidateGenerator(
        SegmentGenConfig(
            per_feature_error="l2",
            feature_reduce="mean",
            smooth_sigma=2.0,
            peak_prominence=0.0,
            peak_distance=4,
            max_peaks=12,
            lengths=(8, 12, 16, 24, 32),
            allow_multi_resolution=True,
            min_len=4,
            max_len=WINDOW_SIZE,
            topk_per_peak=10,
            global_topk=50,
            dedup_iou=0.9,
            w_mass=1.0,
            w_peak=0.25,
            w_compact=0.25,
        )
    )

    class SegmentGeneratorWithRecon:
        def __init__(self, gen, reconstructor_fn):
            self.gen = gen
            self.reconstructor = reconstructor_fn

        def generate(self, x):
            return self.gen.generate(x=x, reconstructor=self.reconstructor)

    segment_generator = SegmentGeneratorWithRecon(base_seg_gen, reconstructor)

    # Part 2: Donor Matcher
    donor_matcher = NormalCoreDonorMatcher(
        normal_core=normal_windows,
        cfg=DonorMatchConfig(
            metric="euclidean",
            standardize=True,
            max_donors=cfg.comte_max_donors,
            max_scan_per_donor=None,
        ),
    )

    # Part 3: Substitutor
    substitutor = SegmentSubstitutor(
        SubstitutionConfig(
            boundary_smoothing=True,
            smoothing_alpha=cfg.comte_smoothing_alpha,
            immutable_sensors=None,
            clamp_min=None,
            clamp_max=None,
        )
    )

    # Part 4: Constraint Evaluator
    constraint_evaluator = CoMTEConstraintEvaluator(
        normal_core=normal_windows.detach().cpu(),
        cfg=CoMTEConstraintConfig(
            sensor_min=float(X_train.min()),
            sensor_max=float(X_train.max()),
            immutable_sensors=None,
            max_delta=None,
            pca_components=5,
        ),
    )

    # Part 5: Selector
    selector = CoMTESelector(
        CoMTESelectionConfig(
            tau=threshold,
            mode="multi",
        ),
    )

    # Part 6: Assemble
    return CoMTEReconstructionCF(
        segment_generator=segment_generator,
        donor_matcher=donor_matcher,
        substitutor=substitutor,
        constraint_evaluator=constraint_evaluator,
        selector=selector,
        failure_handler=FailureHandler(),
        reconstruction_score=reconstruction_score,
        tau=threshold,
        normal_core=normal_windows,
    )


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================
# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    n_test_anomalies: int = 10
    timeout_per_method: float = 60.0  # seconds
    retry_on_failure: bool = False
    save_intermediate: bool = True
    verbose: bool = True


class ResultNormalizer:
    """Normalizes different result formats to a consistent structure."""

    @staticmethod
    def normalize(result: Any) -> Optional[Dict[str, Any]]:
        """
        Convert various result formats to standardized dict.

        Handles:
        - None (failure)
        - Dict with 'xcf' or 'x_cf'
        - Dataclass with attributes

        Returns:
            Dict with keys: x_cf, xcf, score, meta
        """
        if result is None:
            return None

        # Already a dict
        if isinstance(result, dict):
            x_cf = result.get("xcf") or result.get("x_cf")
            if x_cf is None:
                return None

            # Ensure CPU tensor
            if isinstance(x_cf, torch.Tensor):
                x_cf = x_cf.cpu()

            return {
                "x_cf": x_cf,
                "xcf": x_cf,  # Duplicate for compatibility
                "score": result.get("score"),
                "meta": result.get("meta", {}),
            }

        # Dataclass with __dataclass_fields__
        if hasattr(result, "__dataclass_fields__"):
            x_cf = getattr(result, "xcf", None) or getattr(result, "x_cf", None)
            if x_cf is None:
                return None

            if isinstance(x_cf, torch.Tensor):
                x_cf = x_cf.cpu()

            return {
                "x_cf": x_cf,
                "xcf": x_cf,
                "score": getattr(result, "score", None),
                "meta": getattr(result, "meta", {}),
            }

        # Unknown format
        warnings.warn(f"Unknown result type: {type(result).__name__}")
        return None


# ============================================================================
# BENCHMARK EXECUTION WITH TIMING
# ============================================================================


def run_benchmark(
    all_methods: Dict[str, Any],
    X: np.ndarray,
    y_labels: np.ndarray,
    threshold: float,
    normal_windows: torch.Tensor,
    model: ReconstructionAnomalyModule,
    device: torch.device,
    cfg: Optional[BenchmarkConfig] = None,
) -> Tuple[Dict, Dict, Dict, np.ndarray]:  # ✅ Added timing dict
    """
    Run all methods on test anomalies with progress tracking and timing.

    Returns:
        (all_results, all_metrics, all_timings, test_indices)
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    # Select test anomalies
    anomaly_indices = np.where(y_labels == 1)[0]

    if len(anomaly_indices) == 0:
        raise ValueError("No anomalies found in dataset")

    if len(anomaly_indices) < cfg.n_test_anomalies:
        warnings.warn(
            f"Only {len(anomaly_indices)} anomalies available "
            f"(requested {cfg.n_test_anomalies})"
        )
        test_subset = anomaly_indices
    else:
        test_subset = np.random.choice(
            anomaly_indices, cfg.n_test_anomalies, replace=False
        )
    # UPDATED TOP-K SELECTION:
    # Get scores for all anomalies to find the "top" ones
    anomaly_scores = []
    for idx in anomaly_indices:
        anomaly_scores.append(model.score(torch.tensor(X[idx]).to(device)))

    # Sort anomaly indices by their reconstruction scores in descending order
    sorted_indices = [
        x for _, x in sorted(zip(anomaly_scores, anomaly_indices), reverse=True)
    ]
    test_subset = np.array(sorted_indices[: cfg.n_test_anomalies])
    # Initialize storage
    all_results = {name: [] for name in all_methods.keys()}
    all_metrics = {name: [] for name in all_methods.keys()}
    all_timings = {name: [] for name in all_methods.keys()}  # ✅ NEW

    # Metrics calculator
    metrics_calculator = CounterfactualMetrics(MetricsConfig())
    normalizer = ResultNormalizer()

    # Progress tracking
    if cfg.verbose:
        print(f"\n{'=' * 80}")
        print(f"Running benchmark on {len(test_subset)} anomalies...")
        print(f"{'=' * 80}\n")

    # Main benchmark loop
    for i, anom_idx in enumerate(test_subset):
        x_anom = torch.tensor(X[anom_idx], dtype=torch.float32).to(device)

        if cfg.verbose:
            print(f"[{i + 1}/{len(test_subset)}] Anomaly index {anom_idx}")

        for method_name, method in all_methods.items():
            raw_result = None
            error_msg = None

            try:
                # ✅ TIME THE GENERATION
                start_time = time.time()
                raw_result = method.generate(x_anom)
                elapsed_time = time.time() - start_time

                # Store timing
                all_timings[method_name].append(elapsed_time)

                # Normalize result format
                result = normalizer.normalize(raw_result)
                all_results[method_name].append(result)

                if result is not None:
                    # Compute metrics
                    metrics = metrics_calculator.compute(
                        x=x_anom.cpu(),
                        cf_result=result,
                        threshold=threshold,
                        normal_core=normal_windows.cpu(),
                    )
                    all_metrics[method_name].append(metrics)

                    if cfg.verbose:
                        print(
                            f"  {method_name:25s} ✓ Score={result['score']:.4f} ({elapsed_time:.2f}s)"
                        )
                else:
                    all_metrics[method_name].append(None)
                    if cfg.verbose:
                        print(f"  {method_name:25s} ✗ Failed ({elapsed_time:.2f}s)")

            except KeyboardInterrupt:
                print("\n⚠️  Benchmark interrupted by user")
                raise

            except Exception as e:
                # Record failed timing
                elapsed_time = (
                    time.time() - start_time if "start_time" in locals() else 0.0
                )
                all_timings[method_name].append(elapsed_time)

                error_msg = str(e)
                all_results[method_name].append(None)
                all_metrics[method_name].append(None)

                if cfg.verbose:
                    short_error = (
                        error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                    )
                    print(
                        f"  {method_name:25s} ✗ Error: {short_error} ({elapsed_time:.2f}s)"
                    )

            # Memory cleanup
            if raw_result is not None:
                del raw_result
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if cfg.verbose:
            print()

    if cfg.verbose:
        print(f"{'=' * 80}")
        print("Benchmark complete!")
        print(f"{'=' * 80}\n")

    return all_results, all_metrics, all_timings, test_subset


# ============================================================================
# RESULTS ANALYSIS WITH TIMING
# ============================================================================


def analyze_results(
    all_metrics: Dict[str, List[Dict]],
    all_timings: Dict[str, List[float]],  # ✅ NEW
    all_methods: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze and display comprehensive comparison table with timing.
    """
    print(f"\n{'=' * 80}")
    print("Analyzing results...")
    print(f"{'=' * 80}\n")

    # Compute summary statistics for each method
    summary_stats = {}
    for method_name, metrics_list in all_metrics.items():
        valid_metrics = [m for m in metrics_list if m is not None]
        summary_stats[method_name] = summarize_metrics(valid_metrics)

        # ✅ Add timing statistics
        timings = all_timings.get(method_name, [])
        if timings:
            summary_stats[method_name]["timing"] = {
                "mean": float(np.mean(timings)),
                "median": float(np.median(timings)),
                "min": float(np.min(timings)),
                "max": float(np.max(timings)),
                "total": float(np.sum(timings)),
            }

    # ✅ Build CLEAN comparison table (removed redundant columns)
    comparison_data = []
    for method_name, stats in summary_stats.items():
        row = {
            "Method": method_name,
            "Success": format_metric_value(stats.get("found_rate"), "percent"),
            "Avg Score": format_metric_value(
                stats.get("score_cf", {}).get("mean"), "float"
            ),
            "Avg RMSE": format_metric_value(
                stats.get("dist_rmse", {}).get("mean"), "float"
            ),
            "Sparsity": format_metric_value(
                stats.get("frac_changed", {}).get("mean"), "percent"
            ),
            "NN Dist": format_metric_value(
                stats.get("nn_dist_to_normal_core", {}).get("mean"), "float"
            ),
            "Avg Time (s)": format_metric_value(
                stats.get("timing", {}).get("mean"), "float"
            ),
            "Total Time (s)": format_metric_value(
                stats.get("timing", {}).get("total"), "float"
            ),
        }
        comparison_data.append(row)

    # Create DataFrame
    df_comparison = pd.DataFrame(comparison_data)

    # Sort by success rate (desc), then by score (asc)
    df_comparison["Success_numeric"] = (
        df_comparison["Success"].str.rstrip("%").astype(float)
    )
    df_comparison = df_comparison.sort_values(
        by=["Success_numeric", "Avg Score"], ascending=[False, True]
    )
    df_comparison = df_comparison.drop(columns=["Success_numeric"])

    # Display table
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(df_comparison.to_string(index=False))
    print(f"{'=' * 80}\n")

    # Print method rankings
    _print_method_rankings(summary_stats)

    return df_comparison, summary_stats


def _print_method_rankings(summary_stats: Dict[str, Any]) -> None:
    """Print rankings by different criteria."""
    print("\n" + "=" * 80)
    print("METHOD RANKINGS BY CRITERIA")
    print("=" * 80 + "\n")

    ranking_criteria = [
        ("Success Rate", "found_rate", False, "percent"),
        ("Best Score", "score_cf", True, "float"),
        ("Proximity (RMSE)", "dist_rmse", True, "float"),
        ("Sparsity", "frac_changed", True, "percent"),
        ("Plausibility (NN Dist)", "nn_dist_to_normal_core", True, "float"),
        ("Speed (Avg Time)", "timing", True, "float"),  # ✅ Added timing
    ]

    for criterion_name, key, lower_better, format_type in ranking_criteria:
        print(f"{criterion_name}:")

        # Collect scores
        method_scores = []
        for method_name, stats in summary_stats.items():
            if key in ["found_rate", "valid_rate"]:
                score = stats.get(key)
            elif key == "timing":
                # Special handling for timing dict
                score = stats.get("timing", {}).get("mean")
            else:
                score = stats.get(key, {}).get("mean")

            if score is not None:
                method_scores.append((method_name, score))

        # Sort
        method_scores.sort(key=lambda x: x[1], reverse=not lower_better)

        # Display
        for rank, (method_name, score) in enumerate(method_scores, 1):
            formatted_score = format_metric_value(score, format_type)
            print(f"  {rank}. {method_name:30s} {formatted_score}")

        print()


# ============================================================================
# RESULTS SAVING
# ============================================================================


def save_results(
    df_comparison: pd.DataFrame,
    all_results: Dict,
    all_metrics: Dict,
    summary_stats: Dict,
    test_indices: np.ndarray,
    threshold: float,
    output_dir: str,
) -> None:
    """
    Save all results to multiple formats.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("Saving results...")
    print(f"{'=' * 80}\n")

    # -------------------------------------------------------------------------
    # 1. Save CSV (human-readable)
    # -------------------------------------------------------------------------
    csv_path = os.path.join(output_dir, "benchmark_comparison.csv")
    try:
        df_comparison.to_csv(csv_path, index=False)
        print(f"  ✓ CSV saved: {csv_path}")
    except Exception as e:
        print(f"  ✗ CSV save failed: {e}")

    # -------------------------------------------------------------------------
    # 2. Save Pickle (full data)
    # -------------------------------------------------------------------------
    pkl_path = os.path.join(output_dir, "benchmark_all_results.pkl")
    try:
        import pickle

        results_package = {
            "all_results": all_results,
            "all_metrics": all_metrics,
            "summary_stats": summary_stats,
            "test_indices": test_indices,
            "threshold": threshold,
            "comparison_df": df_comparison,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(results_package, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  ✓ Pickle saved: {pkl_path}")
    except Exception as e:
        print(f"  ✗ Pickle save failed: {e}")

    # -------------------------------------------------------------------------
    # 3. Save JSON (summary only, serializable)
    # -------------------------------------------------------------------------
    json_path = os.path.join(output_dir, "benchmark_summary.json")
    try:

        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif pd.isna(obj):
                return None
            return obj

        json_summary = {
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "n_methods": len(summary_stats),
                "n_test_anomalies": len(test_indices),
                "threshold": float(threshold),
            },
            "summary_statistics": make_serializable(summary_stats),
            "comparison_table": df_comparison.to_dict(orient="records"),
        }

        with open(json_path, "w") as f:
            json.dump(json_summary, f, indent=2)

        print(f"  ✓ JSON saved: {json_path}")
    except Exception as e:
        print(f"  ✗ JSON save failed: {e}")

    print(f"\n{'=' * 80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'=' * 80}\n")


# ============================================================================
# UPDATE MAIN FUNCTION
# ============================================================================


def main():
    """Main benchmark execution pipeline."""

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK - ALL 7 CF METHODS ON ATACAMA DATA")
    print("=" * 80)

    # ✅ TIMING: Track total execution time
    total_start = time.time()

    try:
        # =====================================================================
        # 1. Load Data
        # =====================================================================
        print("\n[1/7] Loading Atacama data...")
        X, y_windows = load_and_preprocess(DATAFILE, WINDOW_SIZE, STEP_SIZE, cache=True)

        if X is None:
            raise RuntimeError("Failed to load data")

        y_labels = prepare_labels(y_windows)
        X = X.astype(np.float32)

        print(f"  ✓ Data shape: {X.shape}")
        print(
            f"  ✓ Normal: {(y_labels == 0).sum()} ({(y_labels == 0).mean() * 100:.1f}%)"
        )
        print(
            f"  ✓ Anomaly: {(y_labels == 1).sum()} ({(y_labels == 1).mean() * 100:.1f}%)"
        )

        # =====================================================================
        # 2. Train Model
        # =====================================================================
        print("\n[2/7] Training LSTM Autoencoder...")
        num_features = X.shape[-1]
        cfg = AnomalyConfig(input_size=num_features)

        train_mask = y_labels == 0
        train_windows = X[train_mask]

        print(f"  Training on {len(train_windows)} normal windows...")
        model = train_model(cfg, train_windows, verbose=True)
        model.eval()

        threshold = model.optimal_thresh.item()
        device = DEVICE
        model = model.to(device)

        print(f"  ✓ Model trained. Threshold: {threshold:.4f}")

        # --- NEW: Save Performance and Model ---
        perf_metrics = model.get_performance_summary()
        perf_path = os.path.join(OUTPUT_DIR, "performance_summary.json")
        with open(perf_path, "w") as f:
            json.dump(perf_metrics, f, indent=4)
        print(f"✓ Performance results saved: {perf_path}")

        ts_path = os.path.join(OUTPUT_DIR, "model_ts.pt")
        model.export_torchscript(ts_path)
        # =====================================================================
        # 3. Build Normal Core
        # =====================================================================
        print("\n[3/7] Building normal core...")
        # Randomly select indices for the normal core (no replacement)
        normal_indices = np.random.choice(
            len(train_windows), N_NORMAL_CORE, replace=False
        )
        # Convert selected windows to a PyTorch tensor and move to target device
        normal_core = torch.tensor(
            train_windows[normal_indices], dtype=torch.float32
        ).to(device)
        print(f"  ✓ Normal core: {normal_core.shape}")

        # =====================================================================
        # 4. Initialize Methods  ⚠️ THIS IS WHERE all_methods IS CREATED
        # =====================================================================
        print("\n[4/7] Loading TorchScript model and initializing methods...")

        ts_model_path = os.path.join(OUTPUT_DIR, "model_ts.pt")
        perf_path = os.path.join(OUTPUT_DIR, "performance_summary.json")

        # Ensure the saved files exist before proceeding
        if not os.path.exists(ts_model_path) or not os.path.exists(perf_path):
            raise FileNotFoundError(f"Missing required model files in {OUTPUT_DIR}")

        # Load the saved threshold independently
        with open(perf_path, "r") as f:
            saved_perf = json.load(f)
            # Extract the threshold from the JSON file
            threshold = float(saved_perf["optimal_threshold"])

        print(f"✓ Loaded independent threshold from JSON: {threshold:.4f}")

        # --- ADAPTER FOR BENCHMARK ---
        # Since the benchmark expects a ReconstructionAnomalyModule with a .score() method,
        # we wrap the TorchScript model in a simple proxy class.
        class TorchScriptProxy:
            def __init__(
                self,
                model,
                threshold,
                device,
                num_layers=2,
                input_size=6,
                hidden_size=128,
            ):
                self.model = model
                # Store threshold as a tensor to mimic the original module's buffer
                self.optimal_thresh = torch.tensor(threshold, device=device)
                self.device = device

                # Mock cfg object to satisfy methods like Latent-Space (CMA-ES)
                @dataclass
                class MockConfig:
                    num_layers: int
                    input_size: int
                    hidden_size: int

                self.cfg = MockConfig(
                    num_layers=num_layers,
                    input_size=input_size,
                    hidden_size=hidden_size,
                )

            @torch.no_grad()
            def score(self, x: Any) -> float:
                # Ensure input is a Tensor
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)

                if x.ndim == 2:
                    x = x.unsqueeze(0)

                x = x.to(self.device)
                # The scripted model expects exactly a Tensor
                x_hat = self.model(x)
                return float(torch.mean((x - x_hat) ** 2).item())

            def __call__(self, x: Any) -> torch.Tensor:
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                return self.model(x.to(self.device))

            def eval(self):
                """Mock eval to satisfy initialization checks in CF methods."""
                self.model.eval()
                return self

            def train(self, mode=True):
                """Mock train to satisfy initialization checks."""
                self.model.train(mode)
                return self

        # =====================================================================
        # 4. Initialize Methods (Independent Load)
        # =====================================================================
        print("\n[4/7] Loading artifacts and initializing methods...")

        ts_model_path = os.path.join(OUTPUT_DIR, "model_ts.pt")
        perf_path = os.path.join(OUTPUT_DIR, "performance_summary.json")

        if not os.path.exists(ts_model_path) or not os.path.exists(perf_path):
            raise FileNotFoundError(f"Required artifacts not found in {OUTPUT_DIR}")

        # Load the saved threshold and architecture from JSON
        with open(perf_path, "r") as f:
            saved_perf = json.load(f)
            threshold = float(saved_perf["optimal_threshold"])
            i_size = int(saved_perf["input_size"])
            h_size = int(saved_perf["hidden_size"])
            # Use default num_layers if not explicitly saved in JSON
            n_layers = 2

        print(f"  ✓ Loaded threshold: {threshold:.4f}")

        # Load the TorchScript model
        loaded_model = torch.jit.load(ts_model_path, map_location=DEVICE)
        loaded_model.eval()

        # Create the independent proxy
        model_proxy = TorchScriptProxy(
            model=loaded_model,
            threshold=threshold,
            device=DEVICE,
            num_layers=n_layers,
            input_size=i_size,
            hidden_size=h_size,
        )

        all_methods = initialize_all_methods(
            model_proxy, threshold, normal_core, DEVICE, train_windows
        )
        all_methods.pop("Generative Infilling", None)
        all_methods.pop("Latent-Space (CMA-ES)", None)
        all_methods.pop("Genetic (NSGA-II)", None)

        # =====================================================================
        # 5. Run Benchmark
        # =====================================================================
        print("\n[5/7] Running benchmark...")
        benchmark_cfg = BenchmarkConfig(
            n_test_anomalies=N_TEST_ANOMALIES,
            save_intermediate=False,  # Faster
            verbose=True,
        )

        # ✅ Capture timings
        all_results, all_metrics, all_timings, test_subset = run_benchmark(
            all_methods=all_methods,
            X=X,
            y_labels=y_labels,
            threshold=threshold,
            normal_windows=normal_core,
            device=device,
            model=model_proxy,  # Explicitly assigned
            cfg=benchmark_cfg,
        )

        # =====================================================================
        # 6. Analyze Results
        # =====================================================================
        print("\n[6/7] Analyzing results...")
        df_comparison, summary_stats = analyze_results(
            all_metrics,
            all_timings,  # ✅ Pass timings
            all_methods,
        )

        # =====================================================================
        # 7. Save Results
        # =====================================================================
        print("\n[7/7] Saving results...")
        save_results(
            df_comparison,
            all_results,
            all_metrics,
            summary_stats,
            test_subset,
            threshold,
            OUTPUT_DIR,
        )

        # ============================================================================
        # [8/7] BATCH VISUALIZATION FOR TOP 20 ANOMALIES
        # ============================================================================
        print("\n[8/7] Generating and saving plots for top 20 anomalies...")
        plot_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # test_subset contains the indices of the top 20 anomalies
        for i, anom_idx in enumerate(test_subset):
            # original anomalous window (L, F)
            x_orig = X[anom_idx]

            for method_name, results in all_results.items():
                # Get result for this specific anomaly and method
                res = results[i]

                if res is not None and res.get("x_cf") is not None:
                    # Ensure x_cf is a numpy array
                    x_cf_plot = res["x_cf"]
                    if hasattr(x_cf_plot, "detach"):  # Handle torch tensors
                        x_cf_plot = x_cf_plot.detach().cpu().numpy()

                    # Extract meta information for the plot if available
                    edit_seg = res.get("meta", {}).get("segment")

                    # Construct a descriptive filename
                    save_filename = f"top_{i + 1:02d}_idx{anom_idx}_{method_name.replace(' ', '_')}.png"
                    save_path = os.path.join(plot_dir, save_filename)

                    # Call your provided plot function
                    plot_counterfactual(
                        x=x_orig,
                        x_cf=x_cf_plot,
                        feature_names=FEATURE_COLUMNS,
                        edit_segment=edit_seg,
                        show_diff=True,
                        title=f"Top {i + 1} Anomaly (Idx: {anom_idx}) - {method_name}\nScore: {res['score']:.4f}",
                        save_path=save_path,
                    )

        print(f"✓ Visualization complete. Plots saved to: {plot_dir}")
        # ✅ Print total runtime
        total_elapsed = time.time() - total_start
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE!")
        print("=" * 80)
        print(
            f"\n⏱️  Total Runtime: {total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)"
        )
        print(f"\nOutput directory: {OUTPUT_DIR}")

        return 0  # Success

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n❌ Benchmark failed with error:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
