"""
part7_atacama_comte_run.py
--------------------------
Runs the CoMTE-style Counterfactual method on Atacama data.
Methodology:
1. Detects the anomalous segment (high error region).
2. Searches the 'Normal Core' (Training Set) for a similar context.
3. Transplants the 'healthy' segment from the donor into the anomaly.
4. Smooths boundaries and selects the best fix.
"""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# from methods.comte.Candidate_Selection_Validation import CandidateSelector
# from methods.comte.Constraint_Evaluation import ConstraintEvaluator
# from methods.comte.Failure_Handling import FailureHandler
# from methods.comte.NormalCore_Donor_Matching import Context, NormalCoreDonorMatcher
# from methods.comte.Segment_Candidate_Generation import SegmentCandidate
# from methods.comte.Segment_Substitution_Engine import SegmentSubstitutor
# from methods.comte_final import CoMTEReconstructionCF

"""
part7_atacama_comte_run.py
--------------------------
Orchestrates the CoMTE Pipeline on Atacama Data.
IMPORTS logic from Parts 1-6 instead of redefining it.
"""

import math
import os
import warnings
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. IMPORT YOUR PIPELINE PARTS (1-6) ---
# Make sure these filenames match exactly what you saved
# from part1_segment_generator import TimeSeriesSegmentGenerator
# from part2_donor_matcher import ContextDonorMatcher
# from part3_substitutor import LinearBlendSubstitutor
# from part4_constraint_evaluator import ConstraintConfig, ConstraintEvaluator
# from part5_failure_handler import (
#     FailureHandler,  # Or part6 if combined  # Or part6 if combined
# )
# from part5_selector import BestScoreSelector
# from part6_final_assembly import CoMTEReconstructionCF
# --- 2. IMPORT UTILS (Metrics & Plotting) ---
from utils.metrics import CounterfactualMetrics, MetricsConfig
from utils.plot_counterfactual import plot_counterfactual

warnings.filterwarnings("ignore")


# =============================================================================
# A. DATA SPLITTING & UTILS
# =============================================================================
def find_split_with_anomalies(labels, train_ratio=0.7, val_ratio=0.15):
    """Ensures anomalies exist in all sets for proper testing."""
    N = len(labels)
    anomaly_indices = np.where(labels == 1)[0]
    n_anoms = len(anomaly_indices)

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
# B. MODEL: TRANSFORMER AUTOENCODER (Atacama Specific)
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


class AtacamaModel(pl.LightningModule):
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = 64
        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos = PositionalEncoding(self.d_model, max_len=seq_len + 10)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.d_model,
                nhead=4,
                dim_feedforward=128,
                batch_first=True,
                dropout=0.1,
            ),
            num_layers=2,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self.d_model,
                nhead=4,
                dim_feedforward=128,
                batch_first=True,
                dropout=0.1,
            ),
            num_layers=2,
        )
        self.output_proj = nn.Linear(self.d_model, input_size)

    def forward(self, x):
        src = self.pos(self.input_proj(x))
        memory = self.encoder(src)
        out = self.decoder(src, memory)
        return self.output_proj(out)

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


# =============================================================================
# C. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 1. LOAD & SPLIT DATA
    # -------------------------------------------------------------------------
    data_path = (
        "/home/gupt_ad/conclusion_work/application/ad-xai_pipeline/ATACAMA/Atacama.pkl"
    )
    print(f"\n📂 Loading Data: {data_path}")

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

    # Split
    train_slice, val_slice, test_slice = find_split_with_anomalies(w_labels)
    X_train_raw = windows[train_slice]
    y_train_raw = w_labels[train_slice]
    X_test = windows[test_slice]
    y_test = w_labels[test_slice]

    # Train on Normals Only
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

    # -------------------------------------------------------------------------
    # 2. TRAIN MODEL
    # -------------------------------------------------------------------------
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
    model.to(device).eval()

    # -------------------------------------------------------------------------
    # 3. CONFIGURE CoMTE PIPELINE (Using UPDATED Part 6)
    # -------------------------------------------------------------------------
    print("\n⚙️  Assembling CoMTE Pipeline...")

    # =====================================================
    # A. NORMAL CORE (Donor Pool)
    # =====================================================
    # Use ONLY normal training windows
    K_MAX = 100  # cap for speed
    normal_core_tensor = torch.tensor(X_train[:K_MAX], dtype=torch.float32).to(
        device
    )  # shape: (K, L, F)

    # =====================================================
    # B. MODEL WRAPPERS
    # =====================================================

    # --- reconstructor: x -> x_hat (USED ONLY BY PART 1) ---
    def reconstructor(x_t: torch.Tensor) -> torch.Tensor:
        """
        Input:  (L,F)
        Output: (L,F)
        """
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)
        with torch.no_grad():
            x_hat = model(x_t.to(device))
        return x_hat.squeeze(0)

    # --- reconstruction score: x -> scalar (USED BY PART 5) ---
    def reconstruction_score(x_t: torch.Tensor) -> float:
        """
        Input:  (L,F)
        Output: float reconstruction error
        """
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)
        with torch.no_grad():
            x_hat = model(x_t.to(device))
        return F.mse_loss(x_hat, x_t.to(device)).item()

    # =====================================================
    # C. CHOOSE τ (threshold) FROM NORMAL DATA
    # =====================================================
    with torch.no_grad():
        scores = []
        for b in torch.tensor(X_train[:512], dtype=torch.float32).to(device).split(64):
            x_hat_b = model(b)
            s = ((x_hat_b - b) ** 2).mean(dim=(1, 2))
            scores.append(s.cpu().numpy())
        scores = np.concatenate(scores)

    tau = float(np.percentile(scores, 95))
    print(f"   τ (threshold) set to {tau:.6f} from normal training windows")

    # =====================================================
    # D. PART 1 — SEGMENT GENERATION (BOUND TO RECONSTRUCTOR)
    # =====================================================
    from methods.comte.Segment_Candidate_Generation import (
        SegmentCandidateGenerator,
        SegmentGenConfig,
    )

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
            max_len=L,
            topk_per_peak=10,
            global_topk=50,
            dedup_iou=0.9,
            w_mass=1.0,
            w_peak=0.25,
            w_compact=0.25,
        )
    )

    # 🔴 IMPORTANT: bind reconstructor OUTSIDE Part 6
    class SegmentGeneratorWithRecon:
        def __init__(self, gen, reconstructor_fn):
            self.gen = gen
            self.reconstructor = reconstructor_fn

        def generate(self, x):
            return self.gen.generate(x=x, reconstructor=self.reconstructor)

    segment_generator = SegmentGeneratorWithRecon(base_seg_gen, reconstructor)

    # =====================================================
    # E. PART 2 — DONOR MATCHING
    # =====================================================
    from methods.comte.NormalCore_Donor_Matching import (
        DonorMatchConfig,
        NormalCoreDonorMatcher,
    )

    donor_matcher = NormalCoreDonorMatcher(
        normal_core=normal_core_tensor,
        cfg=DonorMatchConfig(
            metric="euclidean",  # change to "euclidean" if slow, or dtw
            standardize=True,
            max_donors=5,
            max_scan_per_donor=None,
        ),
    )

    # =====================================================
    # F. PART 3 — SEGMENT SUBSTITUTION
    # =====================================================
    from methods.comte.Segment_Substitution_Engine import (
        SegmentSubstitutor,
        SubstitutionConfig,
    )

    substitutor = SegmentSubstitutor(
        SubstitutionConfig(
            boundary_smoothing=True,
            smoothing_alpha=0.4,
            immutable_sensors=None,  # set if needed
            clamp_min=None,
            clamp_max=None,
        )
    )

    # =====================================================
    # G. PART 4 — CONSTRAINT EVALUATION
    # =====================================================
    from methods.comte.Constraint_Evaluation import (
        ConstraintConfig,
        ConstraintEvaluator,
    )

    constraint_evaluator = ConstraintEvaluator(
        normal_core=normal_core_tensor.detach().cpu(),
        cfg=ConstraintConfig(
            sensor_min=float(np.min(X_train)),
            sensor_max=float(np.max(X_train)),
            immutable_sensors=None,
            max_delta=None,
            pca_components=5,
            pca_ref_samples=512,
            boundary_window=2,
        ),
    )

    # =====================================================
    # H. PART 5 — CANDIDATE SELECTION
    # =====================================================
    from methods.comte.Candidate_Selection_Validation import (
        CandidateSelector,
        SelectionConfig,
    )

    selector = CandidateSelector(
        SelectionConfig(
            mode="multi",  # recommended
            tau=tau,
            max_segments=50,
            max_donors_per_segment=3,
        )
    )

    # =====================================================
    # I. PART 6 — FAILURE HANDLING
    # =====================================================
    from methods.comte.Failure_Handling import FailureHandler

    failure_handler = FailureHandler()

    # =====================================================
    # J. FINAL ASSEMBLY (UPDATED PART 6)
    # =====================================================
    from methods.comte_final import CoMTEReconstructionCF

    pipeline = CoMTEReconstructionCF(
        segment_generator=segment_generator,
        donor_matcher=donor_matcher,
        substitutor=substitutor,
        constraint_evaluator=constraint_evaluator,
        selector=selector,
        failure_handler=failure_handler,
        reconstruction_score=reconstruction_score,
        tau=tau,
        normal_core=normal_core_tensor,
    )

    print("✅ CoMTE pipeline assembled successfully")

    # -------------------------------------------------------------------------
    # 4. RUN ON ANOMALY
    # -------------------------------------------------------------------------
    anomaly_indices = np.where(y_test == 1)[0]
    if len(anomaly_indices) > 0:
        idx = anomaly_indices[0]
        x_target = X_test[idx + 11]
        print(f"\n🔍 Explaining REAL Anomaly (Index {idx + 11})")
    else:
        print("\n⚠️  No anomalies in Test Set. Using Synthetic.")
        x_target = X_test[0].copy()
        x_target[20:30] += 3.0

    original_score = reconstruction_score(torch.tensor(x_target, device=device))
    print(f"   Original Score: {original_score:.6f}")

    # EXECUTE
    # The pipeline expects a Tensor (L, F)
    x_target_tensor = torch.tensor(x_target, dtype=torch.float32).to(device)

    output_dict = pipeline.generate(x_target_tensor)

    # -------------------------------------------------------------------------
    # 5. METRICS & PLOTTING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📈 EVALUATION")
    print("=" * 70 + "\n")

    if output_dict and "x_cf" in output_dict:
        print("✅ COUNTERFACTUAL FOUND")

        # Prepare for Metrics Calculator
        # Ensure format matches what utils.metrics expects
        # (x_cf should be a Tensor)

        if isinstance(output_dict["x_cf"], torch.Tensor):
            output_dict["x_cf"] = output_dict["x_cf"].cpu()
        metrics_calc = CounterfactualMetrics(config=MetricsConfig(nn_metric="rmse"))

        metrics = metrics_calc.compute(
            x=x_target_tensor.cpu(),
            cf_result=output_dict,  # output_dict has 'x_cf', 'score', 'meta'
            threshold=original_score * 0.9,  # Just for reference in metrics
            normal_core=normal_core_tensor.cpu(),
        )

        print(f"   Final Score:      {output_dict['score']:.6f}")
        print(f"   Proximity (RMSE): {metrics['dist_rmse']:.4f}")
        print(f"   Validity:         {metrics['valid']}")
        print(
            f"   Replaced Segment: {output_dict['meta'].get('replaced_segment', 'N/A')}"
        )

        # --- PLOT ---
        output_dir = "experiment_run"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "atacama_comte_plot_11.png")

        plot_counterfactual(
            x=x_target,
            x_cf=output_dict["x_cf"].cpu().numpy(),
            # Convert segment tuple (start, end) to what plot expects if needed
            edit_segment=output_dict["meta"].get("replaced_segment"),
            feature_names=feature_cols,
            title=f"CoMTE Counterfactual (Score: {original_score:.4f} -> {output_dict['score']:.4f})",
            save_path=save_path,
        )
        print(f"💾 Plot saved to: {save_path}")

    else:
        print("❌ FAILED (See pipeline logs/failure handler output)")
