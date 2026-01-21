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

from methods.genetic.optimizer_nsga2 import NSGA2Config, NSGA2Optimizer
from methods.genetic.sensor_constraints import SensorConstraintManager

# --- XAI Imports ---
# Adjust these imports based on your exact folder structure.
# Based on the provided context, we assume the genetic helpers are accessible via methods.genetic.*
from methods.genetic_cf import SegmentCounterfactualProblem
from utils.metrics import CounterfactualMetrics, MetricsConfig
from utils.plot_counterfactual import plot_counterfactual

warnings.filterwarnings("ignore")


# =====================================================
# 1. CONFIG & BOILERPLATE (Same as Example 1-3)
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
        # Store labels if provided, otherwise None
        self.y = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        # Fix: Return a dummy tensor (0.0) if labels are None
        # This prevents the "default_collate: batch must contain tensors... found NoneType" error
        label = self.y[idx] if self.y is not None else torch.tensor(0.0)
        return self.x[idx], label


def create_dataloaders(cfg, windows, labels=None):
    dataset = WindowDataset(windows, labels)
    # Simple split for example purposes
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
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout
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
        self.register_buffer("best_val_f1", torch.tensor(0.0))  # Placeholder
        self._val_scores = []

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> float:
        """
        Compute reconstruction anomaly score (MSE) for a single window.
        Required by SegmentCounterfactualProblem.
        """
        self.eval()  # Ensure evaluation mode

        # Handle shape: (L, F) -> (1, L, F)
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        x_hat = self(x)

        # Compute MSE
        score_val = ((x - x_hat) ** 2).mean()
        return float(score_val.item())

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
            # Simple threshold setting: 95th percentile of validation errors
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
# 2. MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    torch.serialization.add_safe_globals([AnomalyConfig])

    # --- A. Setup Data & Model ---
    print("\n" + "=" * 60)
    print("🚀 SETTING UP MODEL & DATA")
    print("=" * 60)

    cfg = AnomalyConfig(max_epochs=50)  # Short epochs for example
    N, L, Fdim = 2000, 64, 6

    # Generate dummy data
    windows = np.random.randn(N, L, Fdim).astype(np.float32)
    # Inject synthetic anomaly
    anomaly_idx = 0
    windows[anomaly_idx, 20:40, :] += 3.0  # Strong anomaly in middle

    model = train_model(cfg, windows)
    model.eval()
    threshold = model.optimal_thresh.item()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare Normal Core (Reference windows for the genetic algorithm)
    # In practice, pick 100-500 confident normal windows from training set
    normal_indices = np.random.choice(len(windows), 200, replace=False)
    normal_core = torch.tensor(windows[normal_indices], dtype=torch.float32).to(device)

    x_anom = torch.tensor(windows[anomaly_idx], dtype=torch.float32).to(device)

    print(f"✅ Model trained. Threshold: {threshold:.4f}")
    print(
        f"   Anomaly Score (Original): {((x_anom - model(x_anom.unsqueeze(0)).squeeze()) ** 2).mean():.4f}"
    )

    # # --- B. Setup Genetic Method Components ---
    print("\n" + "=" * 60)
    print("🧬 CONFIGURING GENETIC OPTIMIZER (NSGA-II)")
    print("=" * 60)

    # [cite_start]1. Constraint Manager (Physics & Realism) [cite: 348]
    #    Enforces bounds, rate limits, and spectral properties based on normal_core
    constraints = SensorConstraintManager(
        normal_core=normal_core,
        value_quantiles=(0.01, 0.99),
        max_delta_quantile=0.99,
        device=str(device),
    )
    print("   -> Constraint Manager initialized")

    # [cite_start]2. Optimizer (NSGA-II) [cite: 326, 332]
    #    Standard multi-objective evolutionary algorithm
    nsga2_config = NSGA2Config(
        pop_size=2000,  # Population size (keep small for speed in example)
        n_gen=500,  # Generations
        seed=42,
        crossover_prob=0.4,
    )
    optimizer = NSGA2Optimizer(nsga2_config)
    print(
        f"   -> NSGA-II initialized (Pop: {nsga2_config.pop_size}, Gens: {nsga2_config.n_gen})"
    )

    # [cite_start]3. Problem Definition [cite: 229]
    #    Connects the model, data, constraints, and optimizer
    cf_method = SegmentCounterfactualProblem(
        model=model,
        threshold=threshold,
        normal_core=normal_core,
        constraints=constraints,
        optimizer=optimizer,
        device=str(device),
        eps_valid=0.5,  # Strict threshold compliance
    )
    print("   -> Genetic Problem defined")
    # --- B. "SANITY CHECK" SETUP ---
    # print("\n" + "=" * 60)
    # print("🧪 RUNNING SANITY CHECK CONFIGURATION")
    # print("=" * 60)

    # # 1. Disable Constraints (Prevent "Hard Check" failures)
    # #    We set bounds to infinity so NOTHING is rejected.
    # constraints = SensorConstraintManager(
    #     normal_core=normal_core, value_quantiles=(0.0, 1.0), device=str(device)
    # )
    # constraints.val_lo[:] = -float("inf")
    # constraints.val_hi[:] = float("inf")

    # # 2. Fast Optimizer Settings
    # #    Small population runs very fast.
    # nsga2_config = NSGA2Config(
    #     pop_size=10,  # Tiny population
    #     n_gen=5,  # Only 5 generations
    #     seed=42,
    #     crossover_prob=0.9,
    # )
    # optimizer = NSGA2Optimizer(nsga2_config)

    # # 3. "Too Easy" Threshold
    # #    We calculate the current score and ask for just 1% improvement.
    # current_score = float(((x_anom - model(x_anom.unsqueeze(0)).squeeze()) ** 2).mean())
    # easy_threshold = current_score * 0.99  # Require only small improvement

    # print(f"   Current Score: {current_score:.4f}")
    # print(f"   Target Score:  {easy_threshold:.4f} (Relaxed for testing)")

    # cf_method = SegmentCounterfactualProblem(
    #     model=model,
    #     threshold=easy_threshold,
    #     normal_core=normal_core,
    #     constraints=constraints,
    #     optimizer=optimizer,
    #     device=str(device),
    #     eps_valid=0.0,  # Strict compliance to the easy threshold
    # )
    # --- C. Generate Counterfactual ---
    print("\n" + "=" * 60)
    print("🔄 GENERATING COUNTERFACTUAL...")
    print("=" * 60)

    result = cf_method.generate(x_anom)

    if result is None:
        print("❌ Failed to find a valid counterfactual.")
        exit()

    print("✅ Counterfactual found!")
    print(f"   Original Score: {result['meta']['base_score']:.4f}")
    print(f"   CF Score:       {result['score']:.4f}")
    print(f"   Objectives:     {result['meta'].get('objectives', 'N/A')}")

    # --- D. Compute Metrics ---

    metrics_calc = CounterfactualMetrics(MetricsConfig())
    if result is not None:
        result["x_cf"] = result["x_cf"].cpu()
    metrics = metrics_calc.compute(
        x=x_anom.cpu(),
        cf_result=result,
        threshold=threshold,
        normal_core=normal_core.cpu(),
    )

    # --- E. Unified Saving (JSON) ---
    print("\n" + "=" * 60)
    print("💾 SAVING RESULTS")
    print("=" * 60)

    def make_serializable(obj):
        """Recursively convert torch/numpy objects to JSON-friendly types"""
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
        "method_name": "Genetic (NSGA-II)",
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
    json_path = os.path.join(output_dir, "genetic_results.json")

    with open(json_path, "w") as f:
        json.dump(unified_output, f, indent=4)

    print(f"✅ Results saved to: {json_path}")

    # --- F. Visualization ---
    plot_path = os.path.join(output_dir, "genetic_plot.png")
    plot_counterfactual(
        x=x_anom.cpu().numpy(),
        x_cf=result["x_cf"].cpu().numpy(),
        title=f"Genetic Counterfactual (NSGA-II)\nScore: {result['score']:.4f} (Thresh: {threshold:.4f})",
        save_path=plot_path,
    )
    print(f"📊 Plot saved to: {plot_path}")
