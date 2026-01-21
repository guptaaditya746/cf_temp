import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.metrics import precision_recall_curve
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

from methods.segment_substitution import SegmentSubstitutionCounterfactual
from utils.metrics import CounterfactualMetrics, MetricsConfig
from utils.plot_counterfactual import plot_counterfactual


# ----------------------------
# CONFIG
# ----------------------------
@dataclass
class AnomalyConfig:
    # data / runtime
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 10
    seed: int = 42
    num_workers: int = 4
    input_size: int = 6
    # model
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # training policy (reconstruction)
    train_normals_only: bool = True
    use_amp: bool = True

    # misc
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"


# =====================================================
# DATASET (WINDOW-SHAPE INPUT)
# =====================================================
class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None):
        assert windows.ndim == 3, f"windows must be (N,L,F). Got shape={windows.shape}"
        self.x = torch.tensor(windows, dtype=torch.float32)

        if labels is None:
            self.y = None
        else:
            assert len(labels) == len(windows), "labels length mismatch"
            self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Optional[Tensor]]:
        if self.y is None:
            return self.x[idx], None
        return self.x[idx], self.y[idx]


def _chronological_split(
    N: int, train_ratio=0.7, val_ratio=0.15
) -> Tuple[slice, slice, slice]:
    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, N)


def create_dataloaders(
    cfg: AnomalyConfig,
    windows: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, DataLoader]:
    N = windows.shape[0]
    tr, va, te = _chronological_split(N)

    if labels is not None and cfg.train_normals_only:
        train_mask = labels[tr] == 0
        train_windows = windows[tr][train_mask]
        train_labels = labels[tr][train_mask]
        train_ds = WindowDataset(train_windows, train_labels)
    else:
        train_ds = WindowDataset(windows[tr], None if labels is None else labels[tr])

    val_ds = WindowDataset(windows[va], None if labels is None else labels[va])
    test_ds = WindowDataset(windows[te], None if labels is None else labels[te])

    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=pin,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin,
        ),
    }
    return loaders


# =====================================================
# RECONSTRUCTION MODEL: LSTM AUTOENCODER
# =====================================================
class LSTMAutoencoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        _, (h_n, c_n) = self.encoder(x)
        z = h_n[-1]
        dec_in = z.unsqueeze(1).repeat(1, L, 1)
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))
        x_hat = self.proj(dec_out)
        return x_hat


class ReconstructionAnomalyModule(pl.LightningModule):
    def __init__(self, cfg: AnomalyConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.model = LSTMAutoencoder(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

        self.register_buffer("optimal_thresh", torch.tensor(0.0))
        self.register_buffer("best_val_f1", torch.tensor(0.0))

        self._val_scores = []
        self._val_targets = []
        self._test_scores = []
        self._test_targets = []

    @staticmethod
    def _window_mse(x: Tensor, x_hat: Tensor) -> Tensor:
        return ((x - x_hat) ** 2).mean(dim=(1, 2))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        scores = self._window_mse(x, x_hat).detach()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if y is not None:
            self._val_scores.append(scores.cpu())
            self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if len(self._val_scores) == 0:
            return

        scores = torch.cat(self._val_scores).numpy()
        targets = torch.cat(self._val_targets).numpy().astype(int)
        best_thresh, best_f1 = self._optimal_f1_threshold(scores, targets)

        if best_f1 > float(self.best_val_f1.item()):
            self.best_val_f1.fill_(best_f1)
            self.optimal_thresh.fill_(best_thresh)

        self.log("val/best_f1", best_f1, prog_bar=True)
        self.log("val/optimal_thresh", best_thresh, prog_bar=True)
        self.log(
            "val/best_val_f1_so_far", float(self.best_val_f1.item()), prog_bar=True
        )

        self._val_scores.clear()
        self._val_targets.clear()

    @staticmethod
    def _optimal_f1_threshold(
        scores: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, float]:
        precision, recall, thresholds = precision_recall_curve(targets, scores)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        if thresholds.size == 0:
            return float(np.median(scores)), 0.0

        best_idx = int(np.nanargmax(f1[:-1]))
        return float(thresholds[best_idx]), float(f1[best_idx])

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        x_hat = self(x)
        scores = self._window_mse(x, x_hat).detach().cpu()
        self.log("test/recon_score_mean", scores.mean(), on_step=False, on_epoch=True)

        if y is not None:
            self._test_scores.append(scores)
            self._test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if len(self._test_scores) == 0:
            return

        scores = torch.cat(self._test_scores).numpy()
        targets = torch.cat(self._test_targets).numpy().astype(int)
        thresh = float(self.optimal_thresh.item())
        preds = (scores >= thresh).astype(int)

        tp = int(((preds == 1) & (targets == 1)).sum())
        fp = int(((preds == 1) & (targets == 0)).sum())
        fn = int(((preds == 0) & (targets == 1)).sum())

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        self.log("test/threshold_used", thresh)
        self.log("test/f1", f1, prog_bar=True)
        self.log("test/precision", precision)
        self.log("test/recall", recall)
        self.log("test/tp", tp)
        self.log("test/fp", fp)
        self.log("test/fn", fn)

        self._test_scores.clear()
        self._test_targets.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# =====================================================
# TRAINING ENTRYPOINT
# =====================================================
def train(
    cfg: AnomalyConfig, windows: np.ndarray, labels: Optional[np.ndarray] = None
) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    loaders = create_dataloaders(cfg, windows, labels)
    model = ReconstructionAnomalyModule(cfg)

    callbacks = [
        EarlyStopping(
            monitor="val_loss" if labels is None else "val/best_f1",
            patience=cfg.patience,
            mode="min" if labels is None else "max",
            min_delta=1e-4,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=cfg.ckpt_dir,
            monitor="val_loss" if labels is None else "val/best_f1",
            mode="min" if labels is None else "max",
            save_top_k=1,
            save_last=True,
            filename="recon-{epoch:02d}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(refresh_rate=10),
    ]

    loggers = [
        CSVLogger(os.path.join(cfg.log_dir, "csv"), name="recon_anomaly"),
        TensorBoardLogger(os.path.join(cfg.log_dir, "tb"), name="recon_anomaly"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="16-mixed" if (cfg.use_amp and torch.cuda.is_available()) else 32,
        deterministic=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    print("🚀 Training (reconstruction-based) ...")
    trainer.fit(model, loaders["train"], loaders["val"])

    print("🧪 Testing best checkpoint ...")
    trainer.test(model, loaders["test"], ckpt_path="best")

    print(f"✅ Done. ckpts='{cfg.ckpt_dir}', logs='{cfg.log_dir}'")


if __name__ == "__main__":
    # =====================================================
    # 1. GENERATE DATA AND TRAIN MODEL
    # =====================================================
    torch.serialization.add_safe_globals([AnomalyConfig])

    cfg = AnomalyConfig()
    N, L, Fdim = 8000, 64, 6
    windows = np.random.randn(N, L, Fdim).astype(np.float32)
    labels = np.random.binomial(1, 0.05, size=(N,)).astype(np.int64)

    cfg = AnomalyConfig(input_size=Fdim)
    train(cfg, windows, labels)

    # =====================================================
    # 2. LOAD MODEL AND PREPARE CF GENERATION
    # =====================================================
    print("\n" + "=" * 70)
    print("🔍 COUNTERFACTUAL GENERATION - SEGMENT SUBSTITUTION")
    print("=" * 70 + "\n")

    ckpt_path = "checkpoints/last.ckpt"
    model = ReconstructionAnomalyModule.load_from_checkpoint(ckpt_path)
    model.eval()
    threshold = model.optimal_thresh.item()

    # Prepare normal core
    normal_windows = torch.tensor(windows[labels == 0][:200], dtype=torch.float32)

    # Select an anomalous window
    anomaly_idx = np.where(labels == 1)[0][0]
    x = torch.tensor(windows[anomaly_idx], dtype=torch.float32)

    print(f"📊 Anomaly index: {anomaly_idx}")
    print(f"   Threshold: {threshold:.4f}")

    # =====================================================
    # 3. GENERATE COUNTERFACTUAL
    # =====================================================
    from methods.segment_substitution import SegmentSubstitutionCounterfactual

    cf_method = SegmentSubstitutionCounterfactual(
        model=model,
        threshold=threshold,
        normal_windows=normal_windows,
        segment_length=8,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    result = cf_method.generate(x)

    # =====================================================
    # 4. COMPUTE METRICS
    # =====================================================
    print("\n" + "=" * 70)
    print("📈 METRICS EVALUATION")
    print("=" * 70 + "\n")

    metrics_calculator = CounterfactualMetrics(
        config=MetricsConfig(
            eps_change=1e-6,
            bounds_q_lo=0.01,
            bounds_q_hi=0.99,
            nn_metric="rmse",
        )
    )

    metrics = metrics_calculator.compute(
        x=x.cpu(),
        cf_result=result,
        threshold=threshold,
        normal_core=normal_windows.cpu(),
    )

    # =====================================================
    # 5. PRINT RESULTS
    # =====================================================
    if result is None:
        print("❌ No counterfactual found")
    else:
        print("✅ COUNTERFACTUAL FOUND\n")

        print(f"CF Generation:")
        print(f"  Score:         {result['score']:.4f}")
        print(f"  Threshold:     {threshold:.4f}")
        print(
            f"  Edit segment:  {result['edit']}"
        )  # ✅ Fixed: segment_substitution has 'edit'
        print(f"  Method:        {result['meta']['method']}")

        print(f"\nValidity:")
        print(f"  Valid:         {metrics['valid']}")
        print(f"  Delta to thr:  {metrics['delta_score_to_thr']:.4f}")

        print(f"\nProximity (Distance to Original):")
        print(f"  RMSE:          {metrics['dist_rmse']:.4f}")
        print(f"  MAE:           {metrics['dist_mae']:.4f}")
        print(f"  Max Abs:       {metrics['dist_max_abs']:.4f}")

        print(f"\nSparsity (Change Structure):")
        print(f"  Fraction changed:  {metrics['frac_changed']:.2%}")
        print(f"  # Segments:        {metrics['n_segments']}")
        print(f"  Max segment len:   {metrics['max_segment_len']}")
        print(f"  Contiguous:        {metrics['contiguous']}")

        print(f"\nSmoothness:")
        print(f"  L2 1st diff:   {metrics['smooth_l2_d1']:.4f}")
        print(f"  L2 2nd diff:   {metrics['smooth_l2_d2']:.4f}")

        print(f"\nPlausibility (vs Normal Core):")
        print(f"  NN distance:       {metrics['nn_dist_to_normal_core']:.4f}")
        print(
            f"  Bounds violations: {metrics['bounds_violations']} ({metrics['bounds_violation_frac']:.2%})"
        )
        print(f"  Z-score (mean):    {metrics['z_abs_mean']:.4f}")
        print(f"  Z-score (max):     {metrics['z_abs_max']:.4f}")

        # =====================================================
        # 6. PLOT COUNTERFACTUAL
        # =====================================================
        print("\n" + "=" * 70)
        print("📊 VISUALIZATION")
        print("=" * 70 + "\n")

        # Create output directory
        output_dir = "experiment_run"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "segment_substitution_plot.png")

        plot_counterfactual(
            x=x.cpu().numpy(),
            x_cf=result["x_cf"].cpu().numpy(),
            edit_segment=result["meta"][
                "segment"
            ],  # ✅ Fixed: use 'segment' not 'edit'
            feature_names=[f"sensor_{i}" for i in range(x.shape[1])],
            title=f"Segment Substitution Counterfactual\n"
            f"Score={result['score']:.4f}, RMSE={metrics['dist_rmse']:.4f}",
            save_path=save_path,
        )

        print(f"💾 Plot saved to: {save_path}")

        # =====================================================
        # 7. SAVE CONSOLIDATED RESULTS
        # =====================================================
        import json

        # ✅ Fixed: Handle non-serializable types properly
        def make_serializable(obj):
            """Convert numpy/torch types to Python native types"""
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
            else:
                return obj

        combined_results = {
            "config": vars(cfg),
            "model_threshold": float(threshold),
            "counterfactual_result": {
                "score": float(result["score"]),
                "edit_segment": result["edit"],  # tuple (start, end)
                "method": result["meta"]["method"],
                "prototype_index": result["meta"].get("prototype_index"),
            },
            "evaluation_metrics": make_serializable(metrics),
        }

        # Save to JSON
        json_path = os.path.join(output_dir, "segment_substitution_summary.json")
        with open(json_path, "w") as f:
            json.dump(combined_results, f, indent=4)

        print(f"💾 Consolidated results saved to: {json_path}")
        print("\n✅ Integration test complete!")
