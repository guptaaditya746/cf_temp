"""
Standalone script for testing Latent-Space Counterfactual Method on Atacama data.
Uses Transformer Autoencoder (Attention-based) for superior global context modeling.
"""

import dataclasses
import json
import math
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
# 1. UTILS: SPLITTER & DATASET
# =====================================================


def find_split_with_anomalies(labels, train_ratio=0.7, val_ratio=0.15):
    """Smart split ensuring anomalies in all sets."""
    N = len(labels)
    anomaly_indices = np.where(labels == 1)[0]
    n_anoms = len(anomaly_indices)

    print(f"\n📊 ANOMALY SPLIT ANALYSIS")
    if n_anoms < 3:
        print("   ⚠️  Not enough anomalies. Using standard time split.")
        train_end = int(N * train_ratio)
        val_end = int(N * (train_ratio + val_ratio))
    else:
        train_anom_idx = int(n_anoms * train_ratio)
        val_anom_idx = int(n_anoms * (train_ratio + val_ratio))
        train_end = anomaly_indices[train_anom_idx] + 1
        val_end = anomaly_indices[val_anom_idx] + 1

        # Safety bounds
        train_end = max(train_end, int(N * 0.5))
        val_end = max(val_end, train_end + 100)
        val_end = min(val_end, N - 1)

    return slice(0, train_end), slice(train_end, val_end), slice(val_end, N)


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None):
        self.x = torch.tensor(windows, dtype=torch.float32)
        self.y = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], (self.y[idx] if self.y is not None else torch.tensor(0.0))


def create_dataloaders(cfg, windows):
    dataset = WindowDataset(windows)
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    return {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
    }


# =====================================================
# 2. MODEL: TRANSFORMER AUTOENCODER
# =====================================================


class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        return x + self.pe[:, : x.size(1), :]


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        seq_length,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model

        # --- Encoder ---
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Bottleneck: Map aggregated context to latent z
        self.to_latent = nn.Linear(d_model, d_model // 2)

        # --- Decoder ---
        self.from_latent = nn.Linear(d_model // 2, d_model)
        # We need positional encoding for decoder too
        self.pos_decoder = PositionalEncoding(d_model, max_len=seq_length + 10)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, input_size)

    def encode(self, x):
        # x: (B, L, F)
        x_emb = self.input_proj(x) * math.sqrt(self.d_model)
        x_emb = self.pos_encoder(x_emb)

        # Apply Transformer Encoder
        # memory: (B, L, d_model)
        memory = self.transformer_encoder(x_emb)

        # Global Average Pooling to get fixed size latent
        # (B, d_model)
        pooled = memory.mean(dim=1)

        # Project to latent dim
        z = self.to_latent(pooled)
        return z

    def decode(self, z):
        # z: (B, latent_dim)
        # Expand z to sequence length to form the "query" for the decoder
        # Or simply use z as the memory input.
        # Standard AE approach: Project z back to sequence and refine.

        B = z.shape[0]
        L = self.seq_length

        hidden = self.from_latent(z)  # (B, d_model)

        # Repeat hidden state for all timesteps to seed generation
        tgt = hidden.unsqueeze(1).repeat(1, L, 1)  # (B, L, d_model)
        tgt = self.pos_decoder(tgt)

        # In a standard Autoencoder (non-autoregressive), we can pass the same target as memory
        # or use the encoder structure symmetrically.
        # Here we use the decoder layer to refine the features.
        output = self.transformer_decoder(tgt, memory=tgt)  # Self-attention on the seed

        x_hat = self.output_proj(output)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


# =====================================================
# 3. CONFIG & LIGHTNING MODULE
# =====================================================


@dataclass
class AnomalyConfig:
    batch_size: int = 512  # Increased for Transformer efficiency
    max_epochs: int = 50
    patience: int = 8
    lr: float = 1e-4  # Transformers like lower LR
    input_size: int = 6  # Set dynamically
    # Transformer Specifics
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    train_normals_only: bool = True
    num_workers: int = 10


class ReconstructionAnomalyModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = TransformerAutoencoder(
            input_size=cfg.input_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            seq_length=64,  # Placeholder, updated in train
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
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=1e-5)


# =====================================================
# 4. MAIN EXECUTION
# =====================================================


def load_atacama_data(data_path: str, window_size: int = 64, stride: int = 1):
    from pathlib import Path

    import pandas as pd

    print(f"📂 Loading Atacama data...")
    path = Path(data_path)
    if path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_pickle(data_path)

    exclude = ["timestamp", "time", "date", "label", "anomaly", "is_anomaly"]
    feats = [c for c in df.columns if c.lower() not in exclude]
    data = df[feats].values.astype(np.float32)

    # Simple normalization
    data = (data - data.mean(0)) / (data.std(0) + 1e-8)

    # Label
    l_col = next(
        (c for c in ["label", "anomaly", "is_anomaly"] if c in df.columns), None
    )
    lbls = df[l_col].values if l_col else np.zeros(len(df))

    windows, w_labels = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i : i + window_size])
        w_labels.append(int(lbls[i : i + window_size].any()))

    return np.array(windows), np.array(w_labels), {"feature_names": feats}


if __name__ == "__main__":
    torch.serialization.add_safe_globals([AnomalyConfig])

    # 1. SETUP
    ATACAMA_DATA_PATH = (
        "/home/gupt_ad/conclusion_work/application/ad-xai_pipeline/ATACAMA/Atacama.pkl"
    )
    windows, labels, metadata = load_atacama_data(
        ATACAMA_DATA_PATH, window_size=64, stride=32
    )
    N, L, Fdim = windows.shape

    # 2. CONFIG
    cfg = AnomalyConfig(input_size=Fdim)

    # 3. SPLIT
    train_slice, val_slice, test_slice = find_split_with_anomalies(labels)
    X_train_raw = windows[train_slice]
    y_train_raw = labels[train_slice]
    X_test = windows[test_slice]
    y_test = labels[test_slice]

    # Filter normals for training
    X_train = X_train_raw[y_train_raw == 0] if cfg.train_normals_only else X_train_raw
    print(f"🔹 Training on {len(X_train)} windows")

    # 4. TRAIN
    print("🚀 Training Transformer Autoencoder...")
    model = ReconstructionAnomalyModule(cfg)
    model.model.seq_length = L

    # Speed optimizations
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        precision="16-mixed",  # Mixed precision for speed
        callbacks=[EarlyStopping("val_loss", patience=cfg.patience)],
    )

    loaders = create_dataloaders(cfg, X_train)
    trainer.fit(model, loaders["train"], loaders["val"])

    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    threshold = model.optimal_thresh.item()
    print(f"✅ Model Threshold: {threshold:.4f}")

    # 5. SELECT ANOMALY
    test_anoms = np.where(y_test == 1)[0]
    if len(test_anoms) == 0:
        print("⚠️ No anomalies in test set, creating synthetic one.")
        x_anom_np = X_test[0].copy()
        x_anom_np[20:40] += 5.0  # Add anomaly
        x_anom = torch.tensor(x_anom_np, dtype=torch.float32)
    else:
        x_anom = torch.tensor(X_test[test_anoms[0]], dtype=torch.float32)

    # 6. LATENT CF ADAPTERS
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

    # 7. EXECUTE CF
    print("🔄 generating Counterfactual...")

    # Data Constraints
    mins = torch.tensor(X_train.min((0, 1)), dtype=torch.float32)
    maxs = torch.tensor(X_train.max((0, 1)), dtype=torch.float32)
    roc = torch.tensor(
        np.percentile(np.diff(X_train, axis=1), 99, axis=(0, 1)), dtype=torch.float32
    )

    constraint = DecodedConstraintSpec(
        value_min=mins - 0.5, value_max=maxs + 0.5, max_rate_of_change=roc
    )

    runner = LatentSpaceCounterfactual(
        model=model,
        threshold=threshold,
        normal_windows=torch.tensor(X_train[:200]),
        device=model.device,
        encoder=encoder_wrapper,
        decoder=decoder_wrapper,
        constraint_spec=constraint,
        cfg=RunnerConfig(
            mode="scalar_cmaes", latent_eps=0.5, cmaes_cfg=CMAESConfig(max_evals=500)
        ),
    )

    res = runner.generate(x_anom)

    if res:
        plot_counterfactual(
            x_anom.cpu().numpy(),
            res["x_cf"].cpu().numpy(),
            title="Transformer CF",
            save_path="experiment_run/trans_cf.png",
        )

        # Save JSON results
        output = {
            "model": "Transformer Autoencoder",
            "threshold": threshold,
            "cf_score": float(res["score"]),
            "original_score": float(res["meta"].get("score_orig", 0.0)),
            "metrics": CounterfactualMetrics(MetricsConfig()).compute(
                x_anom.cpu(), res, threshold, torch.tensor(X_train[:200])
            ),
        }

        # Helper to serialize numpy/torch types
        def safe_serialize(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        with open("experiment_run/transformer_results.json", "w") as f:
            json.dump(output, f, indent=4, default=safe_serialize)

        print("✅ Results saved.")
