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
    import pandas as pd

    from methods.annealing import SegmentSimulatedAnnealingCF
    from methods.clamp import SegmentClampToNormalBoundsCF
    from methods.genetic import SegmentGeneticBlendCF
    from methods.growing_spheres import SegmentGrowingSpheresCF
    from methods.nearest_prototype import NearestPrototypeCounterfactual
    from methods.segment_substitution import SegmentSubstitutionCounterfactual
    from utils.metrics import CounterfactualMetrics, MetricsConfig, summarize_metrics

    # =====================================================
    # 1. GENERATE DATA AND TRAIN MODEL
    # =====================================================
    print("=" * 70)
    print("🚀 BATCH COUNTERFACTUAL EVALUATION - ALL 6 METHODS")
    print("=" * 70 + "\n")
    torch.serialization.add_safe_globals([AnomalyConfig])
    cfg = AnomalyConfig()
    N, L, Fdim = 8000, 64, 6
    windows = np.random.randn(N, L, Fdim).astype(np.float32)
    labels = np.random.binomial(1, 0.05, size=(N,)).astype(np.int64)

    cfg = AnomalyConfig(input_size=Fdim)
    train(cfg, windows, labels)

    # =====================================================
    # 2. LOAD MODEL AND PREPARE DATA
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 LOADING MODEL AND PREPARING DATA")
    print("=" * 70 + "\n")

    ckpt_path = "checkpoints/last.ckpt"
    model = ReconstructionAnomalyModule.load_from_checkpoint(ckpt_path)
    model.eval()
    threshold = model.optimal_thresh.item()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare normal core
    normal_windows = torch.tensor(windows[labels == 0][:200], dtype=torch.float32)

    # Select multiple anomalies to test
    anomaly_indices = np.where(labels == 1)[0][:10]  # First 10 anomalies

    print(f"✅ Model loaded")
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Testing on {len(anomaly_indices)} anomalies")
    print(f"   Normal core size: {normal_windows.shape[0]}")

    # =====================================================
    # 3. INITIALIZE ALL 6 CF METHODS
    # =====================================================
    print("\n" + "=" * 70)
    print("🔧 INITIALIZING ALL METHODS")
    print("=" * 70 + "\n")

    methods = {
        "Segment Substitution": SegmentSubstitutionCounterfactual(
            model=model,
            threshold=threshold,
            normal_windows=normal_windows,
            segment_length=8,
            device=device,
        ),
        "Nearest Prototype": NearestPrototypeCounterfactual(
            model=model,
            threshold=threshold,
            normal_windows=normal_windows,
            device=device,
        ),
        "Growing Spheres": SegmentGrowingSpheresCF(
            model=model,
            threshold=threshold,
            device=device,
            normal_core=normal_windows,
            max_evals=1500,
            min_seg_len=4,
            max_seg_len=32,
            n_dirs=16,
            n_radii=24,
        ),
        "Simulated Annealing": SegmentSimulatedAnnealingCF(
            model=model,
            threshold=threshold,
            device=device,
            normal_core=normal_windows,
            max_evals=2000,
            min_seg_len=4,
            max_seg_len=32,
            steps=600,
        ),
        "Genetic Algorithm": SegmentGeneticBlendCF(
            model=model,
            threshold=threshold,
            device=device,
            normal_core=normal_windows,
            max_evals=4000,
            min_seg_len=4,
            max_seg_len=32,
            pop_size=32,
            generations=40,
        ),
        "Clamp to Normal Bounds": SegmentClampToNormalBoundsCF(
            model=model,
            threshold=threshold,
            device=device,
            normal_core=normal_windows,
            max_evals=2000,
            min_seg_len=4,
            max_seg_len=64,
        ),
    }

    print(f"✅ Initialized {len(methods)} methods")

    # =====================================================
    # 4. RUN ALL METHODS AND COLLECT RESULTS
    # =====================================================
    print("\n" + "=" * 70)
    print("🔍 RUNNING ALL METHODS ON ANOMALIES")
    print("=" * 70 + "\n")

    metrics_calculator = CounterfactualMetrics(
        config=MetricsConfig(
            eps_change=1e-6,
            bounds_q_lo=0.01,
            bounds_q_hi=0.99,
            nn_metric="rmse",
        )
    )

    # Store results
    all_results = {name: [] for name in methods.keys()}
    all_metrics = {name: [] for name in methods.keys()}

    for idx, anomaly_idx in enumerate(anomaly_indices):
        x_anomaly = torch.tensor(windows[anomaly_idx], dtype=torch.float32)

        print(
            f"\n[{idx + 1}/{len(anomaly_indices)}] Processing anomaly at index {anomaly_idx}"
        )

        for method_name, cf_method in methods.items():
            # Generate counterfactual
            result = cf_method.generate(x_anomaly)
            all_results[method_name].append(result)

            # Ensure result tensors are on CPU
            if result is not None and "x_cf" in result:
                result["x_cf"] = result["x_cf"].cpu()

            # Compute metrics
            metrics = metrics_calculator.compute(
                x=x_anomaly.cpu(),
                cf_result=result,
                threshold=threshold,
                normal_core=normal_windows.cpu(),
            )
            all_metrics[method_name].append(metrics)

            # Quick status
            if result is None:
                print(f"   {method_name:25s}: ❌ No CF found")
            else:
                evals = result["meta"].get("evals", "N/A")
                print(
                    f"   {method_name:25s}: ✅ score={result['score']:.4f}, evals={evals}"
                )

    # =====================================================
    # 5. COMPUTE SUMMARY STATISTICS
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70 + "\n")

    summary_stats = {}
    for method_name, metrics_list in all_metrics.items():
        summary_stats[method_name] = summarize_metrics(metrics_list)

        print(f"\n{method_name}")
        print("-" * 70)
        stats = summary_stats[method_name]

        print(f"  Success Rate: {stats['found_rate']:.1%}")
        print(f"  Valid Rate:   {stats['valid_rate']:.1%}")

        if stats["score_cf"]["mean"] is not None:
            print(f"  Score (mean): {stats['score_cf']['mean']:.4f}")

        if stats["dist_rmse"]["mean"] is not None:
            print(f"  RMSE (mean):  {stats['dist_rmse']['mean']:.4f}")

        if stats["frac_changed"]["mean"] is not None:
            print(f"  Sparsity:     {stats['frac_changed']['mean']:.2%}")

        if stats["evals"]["mean"] is not None:
            print(f"  Evals (mean): {stats['evals']['mean']:.0f}")

    # =====================================================
    # 6. CREATE COMPARISON TABLE
    # =====================================================
    print("\n" + "=" * 70)
    print("📈 COMPARISON TABLE")
    print("=" * 70 + "\n")

    comparison_rows = []
    for method_name, stats in summary_stats.items():
        row = {
            "Method": method_name,
            "Success %": f"{stats['found_rate'] * 100:.1f}"
            if stats["found_rate"]
            else "0.0",
            "Valid %": f"{stats['valid_rate'] * 100:.1f}"
            if stats["valid_rate"]
            else "0.0",
            "Score": f"{stats['score_cf']['mean']:.4f}"
            if stats["score_cf"]["mean"]
            else "-",
            "RMSE": f"{stats['dist_rmse']['mean']:.4f}"
            if stats["dist_rmse"]["mean"]
            else "-",
            "Sparsity %": f"{stats['frac_changed']['mean'] * 100:.1f}"
            if stats["frac_changed"]["mean"]
            else "-",
            "Evals": f"{stats['evals']['mean']:.0f}" if stats["evals"]["mean"] else "-",
            "NN Dist": f"{stats['nn_dist_to_normal_core']['mean']:.4f}"
            if stats["nn_dist_to_normal_core"]["mean"]
            else "-",
        }
        comparison_rows.append(row)

    df_comparison = pd.DataFrame(comparison_rows)
    print(df_comparison.to_string(index=False))

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df_comparison.to_csv("results/cf_methods_comparison.csv", index=False)
    print("\n💾 Saved comparison to 'results/cf_methods_comparison.csv'")

    # =====================================================
    # 7. DETAILED METRICS TABLE
    # =====================================================
    print("\n" + "=" * 70)
    print("📋 SAVING DETAILED METRICS")
    print("=" * 70 + "\n")

    detailed_rows = []
    for method_name, stats in summary_stats.items():
        row = {
            "Method": method_name,
            "found_rate": f"{stats['found_rate']:.3f}" if stats["found_rate"] else "-",
            "valid_rate": f"{stats['valid_rate']:.3f}" if stats["valid_rate"] else "-",
        }

        # Add mean values for key metrics
        for key in [
            "score_cf",
            "dist_rmse",
            "dist_mae",
            "frac_changed",
            "n_segments",
            "smooth_l2_d1",
            "nn_dist_to_normal_core",
            "z_abs_mean",
            "evals",
        ]:
            if key in stats and isinstance(stats[key], dict) and "mean" in stats[key]:
                row[f"{key}_mean"] = (
                    f"{stats[key]['mean']:.4f}" if stats[key]["mean"] else "-"
                )

        detailed_rows.append(row)

    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv("results/cf_methods_detailed.csv", index=False)
    print("💾 Saved detailed metrics to 'results/cf_methods_detailed.csv'")

    # =====================================================
    # 8. VISUALIZE EXAMPLES
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 PLOTTING EXAMPLE COUNTERFACTUALS")
    print("=" * 70 + "\n")

    # Plot first successful result for each method
    example_idx = 0
    x_example = torch.tensor(windows[anomaly_indices[example_idx]], dtype=torch.float32)

    for method_name in methods.keys():
        result = all_results[method_name][example_idx]

        if result is not None:
            metrics = all_metrics[method_name][example_idx]

            print(f"\n{method_name}:")
            print(f"  Valid:      {metrics['valid']}")
            print(f"  Score:      {metrics['score_cf']:.4f}")
            print(f"  RMSE:       {metrics['dist_rmse']:.4f}")
            print(f"  Sparsity:   {metrics['frac_changed']:.2%}")

            # Plot
            plot_counterfactual(
                x=x_example.cpu().numpy(),
                x_cf=result["x_cf"].cpu().numpy(),
                edit_segment=result["meta"].get("segment"),
                feature_names=[f"sensor_{i}" for i in range(x_example.shape[1])],
                title=f"{method_name}\n"
                f"Score={metrics['score_cf']:.4f}, RMSE={metrics['dist_rmse']:.4f}, "
                f"Sparsity={metrics['frac_changed']:.1%}",
                show_diff=True,
            )

    # =====================================================
    # 9. METHOD RANKINGS
    # =====================================================
    print("\n" + "=" * 70)
    print("🏆 METHOD RANKINGS")
    print("=" * 70 + "\n")

    ranking_criteria = {
        "Success Rate": ("found_rate", False),
        "Best Score": ("score_cf", True),
        "Proximity (RMSE)": ("dist_rmse", True),
        "Sparsity": ("frac_changed", True),
        "Plausibility": ("nn_dist_to_normal_core", True),
        "Efficiency": ("evals", True),
    }

    for criterion_name, (key, lower_better) in ranking_criteria.items():
        print(f"\n{criterion_name}:")

        method_scores = []
        for method_name, stats in summary_stats.items():
            if key in ["found_rate", "valid_rate"]:
                score = stats.get(key)
            else:
                score = stats.get(key, {}).get("mean")

            if score is not None:
                method_scores.append((method_name, score))

        # Sort
        method_scores.sort(key=lambda x: x[1], reverse=not lower_better)

        for rank, (method_name, score) in enumerate(method_scores, 1):
            if key in ["found_rate", "valid_rate", "frac_changed"]:
                print(f"  {rank}. {method_name:30s} {score:.2%}")
            else:
                print(f"  {rank}. {method_name:30s} {score:.4f}")

    # =====================================================
    # 10. FINAL SUMMARY
    # =====================================================
    print("\n" + "=" * 70)
    print("✅ BATCH EVALUATION COMPLETE")
    print("=" * 70 + "\n")

    print(f"📊 Summary:")
    print(f"   Total anomalies tested: {len(anomaly_indices)}")
    print(f"   Methods evaluated: {len(methods)}")
    print(f"   Results saved to: results/")
    print(f"\n📁 Output files:")
    print(f"   - results/cf_methods_comparison.csv")
    print(f"   - results/cf_methods_detailed.csv")
    print(f"\n🎉 All done!")
