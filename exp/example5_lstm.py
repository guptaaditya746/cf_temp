"""
Standalone script for testing Latent-Space Counterfactual Method on Atacama data.
Uses LSTM Autoencoder with CMA-ES optimization in latent space.
"""

import dataclasses
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from methods.latent.decoded_constraints import DecodedConstraintSpec
from methods.latent.optimizers import CMAESConfig
from methods.latent.selection_and_validation import SelectionConfig
from methods.latent_cf import RunnerConfig

# Latent method imports
from methods.runner import LatentSpaceCounterfactual

# Utility imports
from utils.metrics import CounterfactualMetrics, MetricsConfig
from utils.plot_counterfactual import plot_counterfactual

warnings.filterwarnings("ignore")


# =====================================================
# 1. CONFIG & BOILERPLATE
# =====================================================
@dataclass
class AnomalyConfig:
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 10
    seed: int = 42
    num_workers: int = 4
    input_size: int = 6
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_normals_only: bool = True
    use_amp: bool = True
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None):
        self.x = torch.tensor(windows, dtype=torch.float32)
        self.y = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        label = self.y[idx] if self.y is not None else torch.tensor(0.0)
        return self.x[idx], label


def create_dataloaders(cfg, windows, labels=None):
    dataset = WindowDataset(windows, labels)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    return {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
    }


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        B, L, _ = x.shape
        _, (h_n, c_n) = self.encoder(x)
        dec_in = h_n[-1].unsqueeze(1).repeat(1, L, 1)
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))
        return self.proj(dec_out)


class ReconstructionAnomalyModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = LSTMAutoencoder(
            cfg.input_size, cfg.hidden_size, cfg.num_layers, cfg.dropout
        )
        self.register_buffer("optimal_thresh", torch.tensor(0.0))
        self.register_buffer("best_val_f1", torch.tensor(0.0))
        self._val_scores = []

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> float:
        """Compute reconstruction anomaly score (MSE) for a single window."""
        self.eval()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        x_hat = self(x)
        score_val = ((x - x_hat) ** 2).mean()
        return float(score_val.item())

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Sklearn-style API for anomaly scoring."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 2:  # Single sample (L, F)
            x = x.unsqueeze(0)  # (1, L, F)

        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            x_hat = self(x)
            scores = ((x - x_hat) ** 2).mean(dim=(1, 2))

        return scores.cpu().numpy()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = F.mse_loss(self(x), x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        scores = ((x - x_hat) ** 2).mean(dim=(1, 2))
        self._val_scores.append(scores)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        if self._val_scores:
            all_scores = torch.cat(self._val_scores)
            thresh = torch.quantile(all_scores, 0.95)
            self.optimal_thresh.fill_(thresh)
            self._val_scores.clear()
            self.log("val/optimal_thresh", thresh)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)


def train_model(cfg, windows):
    model = ReconstructionAnomalyModule(cfg)
    loaders = create_dataloaders(cfg, windows)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(model, loaders["train"], loaders["val"])
    return model


# =====================================================
# 2. LATENT SPACE ENCODER/DECODER FUNCTIONS
# =====================================================
def create_lstm_encoder_decoder(lstm_ae_model, sequence_length, device):
    """
    Create encoder/decoder functions for LSTMAutoencoder.

    Args:
        lstm_ae_model: Trained LSTMAutoencoder instance
        sequence_length: Window length L
        device: torch device
    """
    lstm_ae_model.eval()
    lstm_ae_model.to(device)

    def encoder(x):
        """
        Encode window to latent representation.
        x: (L, F) tensor
        Returns: (hidden_size,) latent vector
        """
        with torch.no_grad():
            if x.ndim == 2:  # (L, F)
                x = x.unsqueeze(0).to(device)  # (1, L, F)

            # Get last hidden state from encoder LSTM
            _, (h_n, c_n) = lstm_ae_model.encoder(x)
            z = h_n[-1]  # (1, hidden_size) or (hidden_size,)

            if z.ndim == 2:
                z = z.squeeze(0)  # (hidden_size,)

            return z.cpu()

    def decoder(z):
        """
        Decode latent representation to window.
        z: (hidden_size,) latent vector
        Returns: (L, F) reconstructed window
        """
        with torch.no_grad():
            if z.ndim == 1:  # (hidden_size,)
                z = z.unsqueeze(0).to(device)  # (1, hidden_size)

            # Prepare decoder input
            dec_in = z.unsqueeze(1).repeat(1, sequence_length, 1)  # (1, L, hidden_size)

            # Decode
            dec_out, _ = lstm_ae_model.decoder(dec_in)
            x_hat = lstm_ae_model.proj(dec_out)  # (1, L, F)

            if x_hat.ndim == 3:
                x_hat = x_hat.squeeze(0)  # (L, F)

            return x_hat.cpu()

    return encoder, decoder


# =====================================================
# ATACAMA DATA LOADING
# =====================================================
def load_atacama_data(data_path: str, window_size: int = 64, stride: int = 1):
    """
    Load and preprocess Atacama telescope data.

    Args:
        data_path: Path to Atacama CSV or pickle file
        window_size: Window length for sliding window
        stride: Stride for sliding window

    Returns:
        windows: (N, L, F) array of windowed data
        labels: (N,) array of window-level labels
        metadata: Dictionary with data info
    """
    from pathlib import Path

    import pandas as pd

    print(f"📂 Loading Atacama data from: {data_path}")

    # Load data based on file extension
    path = Path(data_path)
    if path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif path.suffix in [".pkl", ".pickle"]:
        df = pd.read_pickle(data_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    print(f"   Raw data shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")

    # Identify feature columns (exclude time, labels, etc.)
    exclude_cols = ["timestamp", "time", "date", "label", "anomaly", "is_anomaly"]
    feature_cols = [col for col in df.columns if col.lower() not in exclude_cols]

    # Extract features and labels
    data = df[feature_cols].values.astype(np.float32)

    # Get labels if available
    label_col = None
    for col in ["label", "anomaly", "is_anomaly"]:
        if col in df.columns:
            label_col = col
            break

    if label_col:
        point_labels = df[label_col].values
    else:
        print("   ⚠️  No label column found, assuming all normal")
        point_labels = np.zeros(len(df))

    # Normalize data (Z-score normalization)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0) + 1e-8
    data = (data - data_mean) / data_std

    # Create sliding windows
    windows = []
    window_labels = []

    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i : i + window_size]
        window_label = point_labels[i : i + window_size]

        windows.append(window)
        # Window is anomalous if any point is anomalous
        window_labels.append(int(window_label.any()))

    windows = np.array(windows, dtype=np.float32)
    window_labels = np.array(window_labels, dtype=np.int64)

    metadata = {
        "n_windows": len(windows),
        "window_size": window_size,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "n_anomalies": window_labels.sum(),
        "anomaly_ratio": window_labels.mean(),
        "data_mean": data_mean,
        "data_std": data_std,
    }

    print(f"   ✅ Created {len(windows)} windows")
    print(f"   Window shape: {windows.shape}")
    print(f"   Anomalies: {window_labels.sum()} ({window_labels.mean() * 100:.2f}%)")

    return windows, window_labels, metadata


# =====================================================
# 3. MAIN EXECUTION
# =====================================================
# =====================================================
# MAIN EXECUTION (Updated for Atacama)
# =====================================================
if __name__ == "__main__":
    torch.serialization.add_safe_globals([AnomalyConfig])

    # --- A. Load Atacama Data ---
    print("\n" + "=" * 60)
    print("🚀 LOADING ATACAMA DATA")
    print("=" * 60)

    # **CHANGE THIS PATH TO YOUR ATACAMA DATA FILE**
    ATACAMA_DATA_PATH = "/home/gupt_ad/conclusion_work/application/ad-xai_pipeline/ATACAMA/Atacama.pkl"  # ← UPDATE THIS

    # Alternative paths you might have:
    # ATACAMA_DATA_PATH = "/home/gupt_ad/data/atacama.csv"
    # ATACAMA_DATA_PATH = "/home/gupt_ad/conclusion_work/data/atacama_sensors.pkl"

    # Load data
    windows, labels, metadata = load_atacama_data(
        data_path=ATACAMA_DATA_PATH,
        window_size=64,  # Adjust based on your needs
        stride=32,  # 50% overlap
    )

    N, L, Fdim = windows.shape
    print(f"\n   Dataset summary:")
    print(f"   Total windows: {N}")
    print(f"   Window length: {L}")
    print(f"   Features: {Fdim}")
    print(f"   Feature names: {metadata['feature_names']}")

    # --- B. Setup Model Config ---
    print("\n" + "=" * 60)
    print("🏗️  CONFIGURING MODEL")
    print("=" * 60)

    cfg = AnomalyConfig(
        input_size=Fdim,  # Use actual feature count
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        max_epochs=50,
        batch_size=64,
    )

    # Split data for training (only use normal windows)
    train_size = int(N * 0.7)
    val_size = int(N * 0.15)

    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, N)

    # Filter normal windows for training if train_normals_only=True
    if cfg.train_normals_only:
        train_normal_mask = labels[train_indices] == 0
        train_windows = windows[train_indices][train_normal_mask]
        print(f"   Training on {len(train_windows)} normal windows only")
    else:
        train_windows = windows[train_indices]
        print(f"   Training on {len(train_windows)} windows (normal + anomaly)")

    # --- C. Train Model ---
    print("\n" + "=" * 60)
    print("🎓 TRAINING MODEL")
    print("=" * 60)

    model = train_model(cfg, train_windows)
    model.eval()
    threshold = model.optimal_thresh.item()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"✅ Model trained. Threshold: {threshold:.4f}")

    # --- D. Select Anomaly to Explain ---
    print("\n" + "=" * 60)
    print("🔍 SELECTING ANOMALY FOR EXPLANATION")
    print("=" * 60)

    # Find anomalies in test set
    test_anomaly_mask = labels[test_indices] == 1
    test_anomaly_indices = test_indices[test_anomaly_mask]

    if len(test_anomaly_indices) == 0:
        print("❌ No anomalies found in test set!")
        # Create synthetic anomaly for testing
        anomaly_idx = test_indices[0]
        windows[anomaly_idx] += 3.0  # Inject anomaly
        print("   ⚠️  Created synthetic anomaly for testing")
    else:
        anomaly_idx = test_anomaly_indices[0]  # Take first anomaly
        print(f"   Found {len(test_anomaly_indices)} anomalies in test set")
        print(f"   Using anomaly at index {anomaly_idx}")

    # --- E. Prepare Normal Core & Encode to Latent Space ---
    print("\n" + "=" * 60)
    print("🔧 PREPARING LATENT SPACE COMPONENTS")
    print("=" * 60)

    # Select normal windows from training set for reference
    normal_mask = labels[train_indices] == 0
    normal_windows_all = windows[train_indices][normal_mask]

    # Limit to 500 for efficiency
    n_normal = min(500, len(normal_windows_all))
    normal_windows = torch.tensor(normal_windows_all[:n_normal], dtype=torch.float32)

    print(f"   Using {len(normal_windows)} normal windows as reference")

    # Create encoder/decoder functions
    encoder, decoder = create_lstm_encoder_decoder(
        lstm_ae_model=model.model, sequence_length=L, device=device
    )

    # Encode normal windows to latent space
    print("   Encoding normal core to latent space...")
    with torch.no_grad():
        normalcore_z = torch.stack(
            [encoder(normal_windows[i]) for i in range(normal_windows.shape[0])], dim=0
        )

    print(f"   ✅ Normal core latent shape: {normalcore_z.shape}")

    # --- F. Define Constraints from Atacama Data Statistics ---
    print("\n" + "=" * 60)
    print("📊 COMPUTING DATA-DRIVEN CONSTRAINTS")
    print("=" * 60)

    # Use TRAINING data statistics for constraints
    train_data = windows[train_indices][labels[train_indices] == 0]  # Only normal

    # Calculate realistic constraints
    feature_mins = torch.tensor(train_data.min(axis=(0, 1)), dtype=torch.float32)
    feature_maxs = torch.tensor(train_data.max(axis=(0, 1)), dtype=torch.float32)

    # Add margin (15% for safety)
    margin = 0.15 * (feature_maxs - feature_mins)
    feature_mins -= margin
    feature_maxs += margin

    # Calculate rate of change from normal windows
    diffs = np.abs(np.diff(train_data, axis=1))
    max_roc = torch.tensor(np.percentile(diffs, 99, axis=(0, 1)), dtype=torch.float32)

    constraint_spec = DecodedConstraintSpec(
        value_min=feature_mins,
        value_max=feature_maxs,
        max_rate_of_change=max_roc * 1.5,  # Allow some flexibility
        smoothness_weight=1.0,
        roc_weight=1.0,
    )

    print(f"   Feature names: {metadata['feature_names']}")
    print(f"   Value ranges:")
    for i, fname in enumerate(metadata["feature_names"]):
        print(f"      {fname:15s}: [{feature_mins[i]:.3f}, {feature_maxs[i]:.3f}]")
    print(f"   Max rate of change: {max_roc.numpy()}")

    # --- D. Configure Latent CF Method ---
    print("\n" + "=" * 60)
    print("⚙️  CONFIGURING LATENT COUNTERFACTUAL METHOD")
    print("=" * 60)

    # Calculate adaptive latent_eps
    latent_std = normalcore_z.std(dim=0).mean().item()
    latent_eps = 2.0 * latent_std

    cfg_latent = RunnerConfig(
        mode="scalar_cmaes",
        eps_validity=0.05,
        latent_eps=latent_eps,
        seed=42,
        cmaes_cfg=CMAESConfig(
            sigma0=0.3 * latent_std, max_evals=1000, stop_on_first_valid=False
        ),
        selection_cfg=SelectionConfig(
            require_validity=True,
            robustness_trials=6,
            robustness_sigma=0.05 * latent_std,
            robustness_valid_frac=0.67,
        ),
    )

    cf_method = LatentSpaceCounterfactual(
        model=model,
        threshold=threshold,
        normal_windows=normal_windows,
        device=device,
        encoder=encoder,
        decoder=decoder,
        constraint_spec=constraint_spec,
        cfg=cfg_latent,
    )

    print(f"   ✅ Latent CF Method initialized")
    print(f"      Latent dimension: {normalcore_z.shape[1]}")
    print(f"      Latent epsilon: {latent_eps:.3f}")
    print(f"      Max evaluations: {cfg_latent.cmaes_cfg.max_evals}")

    # --- E. Generate Counterfactual ---
    print("\n" + "=" * 60)
    print("🔄 GENERATING COUNTERFACTUAL...")
    print("=" * 60)

    x_anom = torch.tensor(windows[anomaly_idx], dtype=torch.float32)

    print(f"   Explaining anomaly at index {anomaly_idx}")
    print(f"   Window shape: {x_anom.shape}")

    result = cf_method.generate(x_anom)

    if result is None:
        print("❌ Failed to find a valid counterfactual.")
        exit()

    print("✅ Counterfactual found!")
    print(f"   Original Score: {result['meta'].get('score_orig', 'N/A'):.4f}")
    print(f"   CF Score:       {result['score']:.4f}")
    print(f"   Threshold:      {threshold:.4f}")
    print(f"   Valid:          {result['score'] <= threshold}")
    print(f"   Evaluations:    {result['meta'].get('evals', 'N/A')}")
    print(f"   Robust:         {result['meta'].get('robust', 'N/A')}")

    # --- F. Compute Metrics ---

    print("\n" + "=" * 60)
    print("📈 COMPUTING EVALUATION METRICS")
    print("=" * 60)

    metrics_calc = CounterfactualMetrics(MetricsConfig())
    metrics = metrics_calc.compute(
        x=x_anom.cpu(),
        cf_result=result,
        threshold=threshold,
        normal_core=normal_windows,
    )

    # Safe printing with type checking
    def safe_print_metric(name, value):
        """Print metric with proper formatting based on type"""
        if isinstance(value, (int, float, np.number)):
            print(f"   {name:12s}: {float(value):.4f}")
        elif isinstance(value, bool):
            print(f"   {name:12s}: {value}")
        else:
            print(f"   {name:12s}: {value}")

    # Print key metrics
    for key in ["proximity", "sparsity", "validity", "realism", "success"]:
        if key in metrics:
            safe_print_metric(key.capitalize(), metrics[key])

    # --- G. Save Results ---
    print("\n" + "=" * 60)
    print("💾 SAVING RESULTS")
    print("=" * 60)

    def make_serializable(obj):
        """Convert torch/numpy objects to JSON-friendly types"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        return obj

    unified_output = {
        "config": vars(cfg),
        "model_threshold": float(threshold),
        "method_name": "Latent Space (CMA-ES)",
        "latent_config": {
            "latent_dim": int(normalcore_z.shape[1]),
            "latent_eps": float(latent_eps),
            "max_evals": int(cfg_latent.cmaes_cfg.max_evals),
        },
        "model_result": {
            "score": float(result["score"]),
            "valid": result["score"] <= threshold,
            "data": {
                "x_original": make_serializable(x_anom),
                "x_cf": make_serializable(result["x_cf"]),
            },
            "meta": make_serializable(result.get("meta", {})),
        },
        "evaluation_metrics": make_serializable(metrics),
    }

    output_dir = "experiment_run"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "latent_results.json")

    with open(json_path, "w") as f:
        json.dump(unified_output, f, indent=4)

    print(f"✅ Results saved to: {json_path}")

    # --- H. Visualization ---
    plot_path = os.path.join(output_dir, "latent_plot.png")
    plot_counterfactual(
        x=x_anom.cpu().numpy(),
        x_cf=result["x_cf"].cpu().numpy(),
        title=f"Latent Space Counterfactual (CMA-ES)\nScore: {result['score']:.4f} (Thresh: {threshold:.4f})",
        save_path=plot_path,
    )
    print(f"📊 Plot saved to: {plot_path}")

    print("\n" + "=" * 60)
    print("✅ LATENT METHOD TEST COMPLETE")
    print("=" * 60)
