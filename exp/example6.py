"""
part7_atacama_real_integration.py
---------------------------------
Orchestrates the REAL Generative Infilling pipeline on Atacama Telescope data.
INTEGRATION: Uses project-standard 'utils.metrics' and 'utils.plot_counterfactual'.
"""

import json
import math
import os
import warnings
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# --- 2. IMPORT GENERATIVE METHOD PARTS ---
from methods.generative.candidate_selection import CandidateSelector, SelectionConfig
from methods.generative.constraint_evaluation import (
    ConstraintConfig,
    ConstraintEvaluator,
)
from methods.generative.failure_handling import FailureHandler
from methods.generative.infilling_engine import (
    InfillingConfig,
    InfillingEngine,
)
from methods.generative.mask_strategy import MaskStrategy, MaskStrategyConfig
from methods.generative_final import GenerativeInfillingCounterfactual

# --- 1. IMPORT YOUR EXISTING UTILS ---
from utils.metrics import CounterfactualMetrics, MetricsConfig
from utils.plot_counterfactual import plot_counterfactual

warnings.filterwarnings("ignore")


# =============================================================================
# A. UTILITIES & DATASET
# =============================================================================
def find_split_with_anomalies(labels, train_ratio=0.7, val_ratio=0.15):
    N = len(labels)
    anomaly_indices = np.where(labels == 1)[0]
    n_anoms = len(anomaly_indices)

    print(f"\n📊 DATA SPLIT ANALYSIS")
    if n_anoms < 3:
        train_end = int(N * train_ratio)
        val_end = int(N * (train_ratio + val_ratio))
    else:
        train_anom_idx = int(n_anoms * train_ratio)
        val_anom_idx = int(n_anoms * (train_ratio + val_ratio))
        train_end = anomaly_indices[train_anom_idx] + 1
        val_end = anomaly_indices[val_anom_idx] + 1
        train_end = max(train_end, int(N * 0.5))
        val_end = max(val_end, train_end + 100)
        val_end = min(val_end, N - 1)

    return slice(0, train_end), slice(train_end, val_end), slice(val_end, N)


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None):
        self.x = torch.tensor(windows, dtype=torch.float32)
        self.y = (
            torch.tensor(labels, dtype=torch.float32)
            if labels is not None
            else torch.zeros(len(windows))
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# =============================================================================
# B. MODEL: TRANSFORMER AUTOENCODER
# =============================================================================
class PositionalEncoding(nn.Module):
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
        return x + self.pe[:, : x.size(1), :]


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_size, seq_length, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_length + 10)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1
            ),
            num_layers=2,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1
            ),
            num_layers=2,
        )
        self.output_proj = nn.Linear(d_model, input_size)

    def forward(self, x):
        src = self.pos(self.input_proj(x))
        memory = self.encoder(src)
        out = self.decoder(src, memory)
        return self.output_proj(out)


class AtacamaModel(pl.LightningModule):
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerAutoencoder(input_size, seq_len)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        loss = F.mse_loss(self(batch[0]), batch[0])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = F.mse_loss(self(batch[0]), batch[0])
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


class TransformerInfiller:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, x_masked, mask):
        return self.model(x_masked.to(self.device))


# =============================================================================
# C. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    pl.seed_everything(42)

    # 1. LOAD DATA
    # ---------------------------------------------------------
    data_path = (
        "/home/gupt_ad/conclusion_work/application/ad-xai_pipeline/ATACAMA/Atacama.pkl"
    )
    print(f"\n📂 Loading Atacama Data: {data_path}")

    df = pd.read_pickle(data_path)
    exclude_cols = ["timestamp", "time", "date", "label", "anomaly", "is_anomaly"]
    feature_cols = [c for c in df.columns if c.lower() not in exclude_cols]

    raw_data = df[feature_cols].values.astype(np.float32)
    label_col = next(
        (c for c in ["label", "anomaly", "is_anomaly"] if c in df.columns), None
    )
    point_labels = (
        df[label_col].values.astype(int) if label_col else np.zeros(len(df), dtype=int)
    )

    # Normalize
    data_mean = raw_data.mean(axis=0)
    data_std = raw_data.std(axis=0) + 1e-8
    data_norm = (raw_data - data_mean) / data_std

    # Windowing
    L = 64
    stride = 32
    windows = []
    w_labels = []

    for i in range(0, len(data_norm) - L, stride):
        windows.append(data_norm[i : i + L])
        w_labels.append(1 if point_labels[i : i + L].sum() > 0 else 0)
    windows = np.array(windows)
    w_labels = np.array(w_labels)

    # 2. SPLIT & TRAIN
    # ---------------------------------------------------------
    train_slice, val_slice, test_slice = find_split_with_anomalies(w_labels)
    X_train_raw = windows[train_slice]
    y_train_raw = w_labels[train_slice]
    X_test = windows[test_slice]
    y_test = w_labels[test_slice]

    # Filter Training Data (Normals Only)
    X_train = X_train_raw[y_train_raw == 0]
    print(f"   Training on {len(X_train)} normal windows")

    train_loader = DataLoader(
        WindowDataset(X_train, np.zeros(len(X_train))), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        WindowDataset(windows[val_slice], w_labels[val_slice]),
        batch_size=64,
        shuffle=False,
    )

    print("\n🎓 Training Transformer Model...")
    model = AtacamaModel(input_size=windows.shape[2], seq_len=L)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
    )
    trainer.fit(model, train_loader, val_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 3. CONFIGURE PIPELINE
    # ---------------------------------------------------------
    print("\n⚙️  Initializing Generative Pipeline...")
    mask_cfg = MaskStrategyConfig(
        # segment extraction
        top_k=3,
        min_len=4,
        max_len=24,
        merge_gap=2,
        robust_quantile=0.90,
        smooth_sigma=1.0,
        # mask growth
        expand_steps=(0, 2, 4, 6),
        edge_expand_bias=True,
        max_candidates=32,
        # feature handling
        per_feature_mode=False,  # Atacama sensors are coupled → mask all features
        top_feat_k=3,
        # reproducibility
        random_seed=42,
    )

    mask_strategy = MaskStrategy(mask_cfg)
    infill_cfg = InfillingConfig(
        device=str(device),
        dtype=torch.float32,
        mask_token_mode="zeros",  # safest with transformer AE
        n_samples=6,  # small stochastic ensemble
        sampling_seed=42,
        deterministic=False,
        smooth_masked=True,
        smooth_sigma=1.0,
        clamp_output=False,
    )

    infilling_engine = InfillingEngine(
        infiller=TransformerInfiller(model, device),
        cfg=infill_cfg,
        normalcore_feature_mean=windows.mean(axis=(0, 1)),
    )
    con_cfg = ConstraintConfig(
        # hard constraints
        value_min=float(windows.min()),
        value_max=float(windows.max()),
        # max_abs_delta=2.0,  # allow correction but not teleportation
        # immutable_features=None,
        # # soft constraints
        # max_rate_of_change=3.0,
        # curvature_weight=1.0,
        # spectral_weight=0.5,
        # coupling_weight=1.0,
        # dtw_weight=1.0,
        # # signal processing
        # savgol_window=7,
        # savgol_poly=2,
        # psd_nperseg=32,
        # # coupling
        # pca_components=min(3, windows.shape[2]),
    )

    constraint_evaluator = ConstraintEvaluator(
        con_cfg,
        normalcore=windows[:100],  # small clean reference slice
    )
    sel_cfg = SelectionConfig(
        threshold=0.0,  # overwritten dynamically in generate()
        mode="lexicographic",
        # single-objective fallback
        w_score_excess=10.0,
        w_mask_size=1.0,
        w_soft_penalty=1.0,
        lexicographic_order=(
            "score_excess",
            "mask_size",
            "soft_penalty",
        ),
        allow_equal_threshold=True,
    )

    candidate_selector = CandidateSelector(
        cfg=sel_cfg,
        score_fn=None,  # injected at runtime
    )

    # 4. RUN ON ANOMALY
    # ---------------------------------------------------------
    anomaly_indices = np.where(y_test == 1)[0]
    if len(anomaly_indices) > 0:
        idx = anomaly_indices[0]
        x_target = X_test[idx + 2]
        print(f"\n🔍 Explaining REAL Anomaly (Index {idx + 2})")
    else:
        print("\n⚠️  No anomalies in Test Set. Using Synthetic.")
        x_target = X_test[0].copy()
        x_target[20:30] += 3.0

    # Scorer
    def anomaly_scorer(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            x_hat = model(x_t)
        return F.mse_loss(x_hat, x_t).item()

    # Get Reconstruction Profile
    x_t_target = torch.tensor(x_target).unsqueeze(0).to(device)
    with torch.no_grad():
        e_t = (
            torch.mean((x_t_target - model(x_t_target)) ** 2, dim=2)
            .squeeze()
            .cpu()
            .numpy()
        )

    original_score = anomaly_scorer(x_target)
    threshold = original_score * 0.6

    # Instantiate Pipeline Orchestrator (Part 6)
    pipeline = GenerativeInfillingCounterfactual(
        mask_strategy=mask_strategy,
        infilling_engine=infilling_engine,
        constraint_evaluator=constraint_evaluator,
        candidate_selector=candidate_selector,
        failure_handler=FailureHandler(),
    )

    # EXECUTE
    cf_result_dataclass = pipeline.generate(
        x=x_target,
        reconstruction_error_t=e_t,
        anomaly_score_fn=anomaly_scorer,
        threshold=threshold,
    )

    # =========================================================================
    # 5. METRICS & VISUALIZATION (USING YOUR EXISTING UTILS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("📈 METRICS EVALUATION")
    print("=" * 70 + "\n")

    if cf_result_dataclass is not None:
        print("✅ COUNTERFACTUAL FOUND")

        # --- A. ADAPTER: Convert Dataclass -> Dict for your Metric Utility ---
        # Your metrics calculator expects a dictionary with keys like 'x_cf', 'meta', etc.
        # But our pipeline returns a strict Dataclass. We convert it here.
        cf_result_dict = {
            "x_cf": torch.tensor(cf_result_dataclass.x_cf, dtype=torch.float32),
            "score": cf_result_dataclass.score,
            "meta": cf_result_dataclass.meta,
        }

        # --- B. COMPUTE METRICS ---
        metrics_calculator = CounterfactualMetrics(
            config=MetricsConfig(
                eps_change=1e-6,
                bounds_q_lo=0.01,
                bounds_q_hi=0.99,
                nn_metric="rmse",
            )
        )

        # We need the Normal Core (reference normals) as Tensor on CPU
        normal_core_tensor = torch.tensor(X_train[:200], dtype=torch.float32)

        metrics = metrics_calculator.compute(
            x=torch.tensor(x_target, dtype=torch.float32),
            cf_result=cf_result_dict,
            threshold=threshold,
            normal_core=normal_core_tensor,
        )

        # Print Key Metrics
        print(
            f"  Score:             {cf_result_dataclass.score:.4f} (Threshold: {threshold:.4f})"
        )
        print(f"  Validity:          {metrics['valid']}")
        print(f"  Proximity (RMSE):  {metrics['dist_rmse']:.4f}")
        print(f"  Sparsity (Segs):   {metrics['n_segments']}")
        print(f"  Plausibility (NN): {metrics['nn_dist_to_normal_core']:.4f}")

        # --- C. PLOT (Using utils.plot_counterfactual) ---
        print("\n" + "=" * 70)
        print("📊 VISUALIZATION")
        print("=" * 70 + "\n")

        output_dir = "experiment_run"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "atacama_generative_plot.png")

        plot_counterfactual(
            x=x_target,
            x_cf=cf_result_dataclass.x_cf,
            # We don't have a single 'edit_segment' because Generative Infilling might
            # change multiple non-contiguous regions. We leave it None to show full comparison.
            edit_segment=None,
            feature_names=feature_cols,
            title=f"Generative Infilling (Atacama)\nScore: {original_score:.4f} -> {cf_result_dataclass.score:.4f}",
            save_path=save_path,
        )
        print(f"💾 Plot saved to: {save_path}")

        # --- D. SAVE JSON (Consolidated) ---
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
            return obj

        combined_results = {
            "model": "Transformer Autoencoder",
            "model_threshold": float(threshold),
            "counterfactual_result": make_serializable(cf_result_dict),
            "evaluation_metrics": make_serializable(metrics),
        }

        json_path = os.path.join(output_dir, "atacama_results_consolidated.json")
        with open(json_path, "w") as f:
            json.dump(combined_results, f, indent=4)
        print(f"💾 Results saved to: {json_path}")

    else:
        print("\n❌ FAILED to find valid counterfactual.")
