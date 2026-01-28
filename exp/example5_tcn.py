"""
Standalone script for testing Latent-Space Counterfactual Method on Atacama data.
Uses TCN Autoencoder (via pytorch-tcn) with Smart Anomaly Splitting.
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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset

# --- EXTERNAL LIBRARY CHECK ---
try:
    from pytorch_tcn import TCN
except ImportError:
    raise ImportError("Please install the external library: pip install pytorch-tcn")

from methods.latent.decoded_constraints import DecodedConstraintSpec
from methods.latent.latent_cf import RunnerConfig
from methods.latent.optimizers import CMAESConfig
from methods.latent.selection_and_validation import SelectionConfig

# Latent method imports
from methods.latent_final import LatentSpaceCounterfactual

# Utility imports
from utils.metrics import CounterfactualMetrics, MetricsConfig
from utils.plot_counterfactual import plot_counterfactual

warnings.filterwarnings("ignore")


# =====================================================
# 1. DATA PREPROCESSING & SPLITTING UTILS
# =====================================================


def find_split_with_anomalies(labels, train_ratio=0.7, val_ratio=0.15):
    """
    Find split points that ensure anomalies are distributed across Train, Val, and Test.
    Uses anomaly index percentiles to guarantee distribution.
    """
    N = len(labels)
    anomaly_indices = np.where(labels == 1)[0]
    n_anoms = len(anomaly_indices)

    print(f"\n📊 ANOMALY SPLIT ANALYSIS")
    print(f"   Total anomalies: {n_anoms}")

    if n_anoms < 3:
        print(
            "   ⚠️  Not enough anomalies to guarantee split. Using standard time split."
        )
        train_end = int(N * train_ratio)
        val_end = int(N * (train_ratio + val_ratio))
    else:
        print(f"   First anomaly at index: {anomaly_indices[0]}")
        print(f"   Last anomaly at index: {anomaly_indices[-1]}")

        # Find indices in the *anomaly list* corresponding to the ratios
        # e.g., if we have 100 anomalies, we cut at index 70 and 85
        train_anom_idx = int(n_anoms * train_ratio)
        val_anom_idx = int(n_anoms * (train_ratio + val_ratio))

        # Map back to time-series indices
        # We add some buffer after the anomaly to include it in the previous set
        train_end = anomaly_indices[train_anom_idx] + 1
        val_end = anomaly_indices[val_anom_idx] + 1

        # Safety checks to ensure we don't go out of bounds or overlap strangely
        train_end = max(train_end, int(N * 0.5))  # Ensure train is at least 50% of data
        val_end = max(val_end, train_end + 100)  # Ensure val has some size
        val_end = min(val_end, N - 1)

    # Verify each split has anomalies
    train_anoms = np.sum((anomaly_indices >= 0) & (anomaly_indices < train_end))
    val_anoms = np.sum((anomaly_indices >= train_end) & (anomaly_indices < val_end))
    test_anoms = np.sum(anomaly_indices >= val_end)

    print(f"   Train: 0 to {train_end} ({train_anoms} anomalies)")
    print(f"   Val:   {train_end} to {val_end} ({val_anoms} anomalies)")
    print(f"   Test:  {val_end} to {N} ({test_anoms} anomalies)")

    return slice(0, train_end), slice(train_end, val_end), slice(val_end, N)


# =====================================================
# 2. CONFIG & BOILERPLATE
# =====================================================
@dataclass
class AnomalyConfig:
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 10
    seed: int = 42
    num_workers: int = 4
    input_size: int = 6
    # TCN Specific Configs
    num_channels: List[int] = dataclasses.field(default_factory=lambda: [64, 128])
    kernel_size: int = 3
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_normals_only: bool = True
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
    # Note: We assume 'windows' passed here is strictly the training set derived from the split
    train_size = int(0.9 * len(dataset))  # Internal validation for overfitting check
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


# =====================================================
# 3. MODEL: TCN AUTOENCODER
# =====================================================
class TCNAutoencoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.encoder_tcn = TCN(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=False,
        )
        self.to_latent = nn.Linear(num_channels[-1], num_channels[-1])
        dec_channels = num_channels[::-1]
        self.from_latent = nn.Linear(num_channels[-1], num_channels[-1])
        self.decoder_tcn = TCN(
            num_inputs=num_channels[-1],
            num_channels=dec_channels + [input_size],
            kernel_size=kernel_size,
            dropout=dropout,
            causal=False,
        )

    def encode(self, x):
        x_tcn = x.permute(0, 2, 1)
        enc_out = self.encoder_tcn(x_tcn)
        z = F.adaptive_avg_pool1d(enc_out, 1).squeeze(-1)
        z = self.to_latent(z)
        return z

    def decode(self, z):
        z = self.from_latent(z)
        z_expanded = z.unsqueeze(-1).repeat(1, 1, self.seq_length)
        dec_out = self.decoder_tcn(z_expanded)
        return dec_out.permute(0, 2, 1)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class ReconstructionAnomalyModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = TCNAutoencoder(
            input_size=cfg.input_size,
            num_channels=cfg.num_channels,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            seq_length=64,  # Placeholder
        )
        self.register_buffer("optimal_thresh", torch.tensor(0.0))
        self._val_scores = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
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
    L = windows.shape[1]
    model = ReconstructionAnomalyModule(cfg)
    model.model.seq_length = L

    loaders = create_dataloaders(cfg, windows)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=cfg.patience, mode="min")
        ],
    )
    trainer.fit(model, loaders["train"], loaders["val"])
    return model


# =====================================================
# 4. DATA LOADING & MAIN
# =====================================================
def load_atacama_data(data_path: str, window_size: int = 64, stride: int = 1):
    from pathlib import Path

    import pandas as pd

    print(f"📂 Loading Atacama data from: {data_path}")
    path = Path(data_path)
    if path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif path.suffix in [".pkl", ".pickle"]:
        df = pd.read_pickle(data_path)
    else:
        raise ValueError("Unsupported format")

    exclude_cols = ["timestamp", "time", "date", "label", "anomaly", "is_anomaly"]
    feature_cols = [col for col in df.columns if col.lower() not in exclude_cols]
    data = df[feature_cols].values.astype(np.float32)

    label_col = next(
        (col for col in ["label", "anomaly", "is_anomaly"] if col in df.columns), None
    )
    point_labels = df[label_col].values if label_col else np.zeros(len(df))

    data_mean, data_std = data.mean(axis=0), data.std(axis=0) + 1e-8
    data = (data - data_mean) / data_std

    windows, window_labels = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i : i + window_size]
        windows.append(window)
        window_labels.append(int(point_labels[i : i + window_size].any()))

    return np.array(windows), np.array(window_labels), {"feature_names": feature_cols}


if __name__ == "__main__":
    torch.serialization.add_safe_globals([AnomalyConfig])

    # 1. Load Data
    ATACAMA_DATA_PATH = (
        "/home/gupt_ad/conclusion_work/application/ad-xai_pipeline/ATACAMA/Atacama.pkl"
    )
    windows, labels, metadata = load_atacama_data(
        ATACAMA_DATA_PATH, window_size=64, stride=32
    )
    N, L, Fdim = windows.shape

    # 2. Config
    cfg = AnomalyConfig(
        input_size=Fdim,
        num_channels=[32, 64, 128],
        kernel_size=5,
        train_normals_only=True,  # We usually train AEs only on normal data
    )

    # 3. SPLIT DATA USING YOUR CUSTOM FUNCTION
    # This returns slices: slice(0, 1000), slice(1000, 1500), etc.
    train_slice, val_slice, test_slice = find_split_with_anomalies(
        labels, train_ratio=0.7, val_ratio=0.15
    )

    # Materialize the data based on slices
    X_train_raw = windows[train_slice]
    y_train_raw = labels[train_slice]

    X_val = windows[val_slice]
    y_val = labels[val_slice]

    X_test = windows[test_slice]
    y_test = labels[test_slice]

    # 4. FILTER TRAIN SET (Constraint Check)
    # The split ensured anomalies exist in the 'train_slice' time range.
    # But for the AE training, we might want to filter them out.
    if cfg.train_normals_only:
        print(f"🧹 Filtering anomalies from training set (train_normals_only=True)")
        train_mask = y_train_raw == 0
        X_train = X_train_raw[train_mask]
        print(
            f"   Original Train Size: {len(X_train_raw)} -> Filtered Size: {len(X_train)}"
        )
    else:
        X_train = X_train_raw

    # 5. Train
    print("\n" + "=" * 30 + " TRAINING " + "=" * 30)
    model = train_model(cfg, X_train)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    threshold = model.optimal_thresh.item()
    print(f"✅ Trained. Threshold: {threshold:.4f}")

    # 6. Select Anomaly for CF
    test_anoms = np.where(y_test == 1)[0]
    if len(test_anoms) > 0:
        # Map relative test index back to absolute window array if needed,
        # but here we just grab from X_test
        x_anom = torch.tensor(X_test[test_anoms[0]], dtype=torch.float32)
        print(f"🔍 Selected anomaly from Test set at relative index {test_anoms[0]}")
    else:
        # Fallback to Val set if test has no anomalies (rare with your splitter)
        val_anoms = np.where(y_val == 1)[0]
        x_anom = torch.tensor(X_val[val_anoms[0]], dtype=torch.float32)
        print(
            f"🔍 Selected anomaly from Val set (fallback) at relative index {val_anoms[0]}"
        )

    # 7. Run Latent CF (Standard Setup)
    # Prepare Encoder/Decoder
    def encoder_wrapper(x):
        with torch.no_grad():
            if x.ndim == 2:
                x = x.unsqueeze(0).to(model.device)
            return model.model.encode(x).squeeze(0).cpu()

    def decoder_wrapper(z):
        with torch.no_grad():
            if z.ndim == 1:
                z = z.unsqueeze(0).to(model.device)
            return model.model.decode(z).squeeze(0).cpu()

    # Normal Core (Reference)
    normal_refs = torch.tensor(X_train[:500], dtype=torch.float32)

    # Constraints
    feature_mins = torch.tensor(X_train.min(axis=(0, 1)), dtype=torch.float32)
    feature_maxs = torch.tensor(X_train.max(axis=(0, 1)), dtype=torch.float32)
    margin = 0.1 * (feature_maxs - feature_mins)

    constraint_spec = DecodedConstraintSpec(
        value_min=feature_mins - margin,
        value_max=feature_maxs + margin,
        max_rate_of_change=torch.tensor(
            np.percentile(np.diff(X_train, axis=1), 99, axis=(0, 1)),
            dtype=torch.float32,
        ),
    )

    # Execute
    print("\n" + "=" * 30 + " GENERATING CF " + "=" * 30)
    runner = LatentSpaceCounterfactual(
        model=model,
        threshold=threshold,
        normal_windows=normal_refs,
        device=model.device,
        encoder=encoder_wrapper,
        decoder=decoder_wrapper,
        constraint_spec=constraint_spec,
        cfg=RunnerConfig(
            latent_eps=1.0, mode="scalar_cmaes", cmaes_cfg=CMAESConfig(max_evals=500)
        ),
    )

    res = runner.generate(x_anom)

    if res:
        plot_counterfactual(
            x_anom.cpu().numpy(),
            res["x_cf"].cpu().numpy(),
            title="TCN Counterfactual",
            save_path="experiment_run/tcn_cf.png",
        plot_counterfactual(
            x_anom.cpu().numpy(),
            res["x_cf"].cpu().numpy(),
            title="TCN Counterfactual",
            save_path="experiment_run/tcn_cf.png",
        )
        plot_counterfactual(x_anom.cpu().numpy(), res['x_cf'].cpu().numpy(), title="TCN Counterfactual", save_path="experiment_run/tcn_cf.png")
        print("✅ Done. Result saved.")
