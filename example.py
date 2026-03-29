#!/usr/bin/env python
# coding: utf-8

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from cftsad import CFFailure, CFResult, CounterfactualExplainer
except Exception:
    from src.cftsad import CFFailure, CFResult, CounterfactualExplainer


DATA_PATH = "./artifacts/ /Atacama.pkl"
VAR_LIST_PATH = "./artifacts/var_list.csv"

WINDOW_SIZE = 24
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
CALIB_RATIO = 0.1

DATA_BATCH_SIZE = 128
NUM_WORKERS = 8
THRESHOLD_PERCENTILE = 99

MODEL_NAME = "lstm_autoencoder"
MODEL_CONFIG = {
    "seq_len": WINDOW_SIZE,
    "n_features": 6,
    "hidden_dim": 32,
    "latent_dim": 8,
    "lr": 1e-3,
}

TRAINER_CONFIG = {
    "max_epochs": 50,
    "devices": 1,
    "log_every_n_steps": 10,
}

CFTSAD_BASE_CONFIG = {
    "use_constraints_v2": True,
    "enable_fallback_chain": False,
    "fallback_retry_budget": 2,
    "normal_core_threshold_quantile": 0.95,
    "normal_core_filter_factor": 1.0,
    "normal_core_max_size": 100,
    "normal_core_use_diversity_sampling": True,
    "random_seed": 42,
}

CFTSAD_METHOD_CONFIGS = {
    "nearest": {
        "nearest_top_k": 10,
        "nearest_alpha_steps": 5,
        "nearest_use_weighted_distance": True,
        "fallback_methods": ("segment", "motif", "genetic"),
    },
    "segment": {
        "segment_smoothing": True,
        "segment_n_candidates": 4,
        "segment_top_k_donors": 8,
        "segment_context_width": 2,
        "segment_crossfade_width": 3,
        "fallback_methods": ("motif", "nearest", "genetic"),
    },
    "motif": {
        "motif_top_k": 5,
        "motif_n_segments": 2,
        "motif_length_factors": (0.75, 1.0, 1.25),
        "motif_context_weight": 0.2,
        "motif_use_affine_fit": True,
        "fallback_methods": ("segment", "nearest", "genetic"),
    },
    "genetic": {
        "population_size": 50,
        "n_generations": 20,
        "use_plausibility_objective": True,
        "structured_mutation_weight": 0.35,
        "top_m_solutions": 5,
        "early_stop_patience": 15,
        "fallback_methods": ("segment", "motif", "nearest"),
    },
}


class UnsupervisedTimeSeriesDataset(Dataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AtacamaDataModule(pl.LightningDataModule):
    def __init__(self, x_train, x_val, x_calib, x_test, batch_size=64, num_workers=2):
        super().__init__()
        self.x_train = x_train
        self.x_val = x_val
        self.x_calib = x_calib
        self.x_test = x_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = UnsupervisedTimeSeriesDataset(self.x_train)
            self.val_dataset = UnsupervisedTimeSeriesDataset(self.x_val)

        if stage == "test" or stage is None:
            self.test_dataset = UnsupervisedTimeSeriesDataset(self.x_test)

        if stage == "predict" or stage is None:
            self.calib_dataset = UnsupervisedTimeSeriesDataset(self.x_calib)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.calib_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class TimeSeriesAutoencoder(pl.LightningModule):
    def __init__(self, seq_len=24, n_features=6, hidden_dim=64, latent_dim=16, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr

        self.encoder_lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.encoder_linear = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.decoder_linear = nn.Linear(self.hidden_dim, self.n_features)
        self.criterion = nn.MSELoss()

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden.squeeze(0)
        latent = self.encoder_linear(hidden)
        return latent

    def decode(self, latent):
        latent_repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_out, _ = self.decoder_lstm(latent_repeated)
        reconstruction = self.decoder_linear(decoder_out)
        return reconstruction

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def training_step(self, batch, batch_idx):
        x_hat = self(batch)
        loss = self.criterion(x_hat, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat = self(batch)
        loss = self.criterion(x_hat, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


MODEL_REGISTRY = {
    "lstm_autoencoder": TimeSeriesAutoencoder,
}


def make_output_dir():
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"results/run_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def configure_runtime():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    warnings.filterwarnings("ignore")
    plt.style.use("ggplot")
    sns.set_theme(style="whitegrid")
    print(f"Currently using: {device}")
    return device


def load_feature_names(path):
    with open(path, "r") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def load_and_clean_dataframe(data_path, feature_names, output_dir):
    df = pd.read_pickle(data_path)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("h")

    df[feature_names] = df[feature_names].ffill(limit=3)
    df[feature_names] = df[feature_names].fillna(df[feature_names].median())

    df["label"] = df["label"].fillna(0).astype(int)
    df.loc[df["precipitation (mm)"] > 2.0, "label"] = 1

    final_columns = feature_names + ["label"]
    df_clean = df[final_columns]
    df_clean.to_pickle(f"{output_dir}/01_ingested_qa_data.pkl")

    print(f"Cleaned Shape: {df_clean.shape}")
    print(f"Schema Locked. Features: {feature_names}")
    return df_clean


def scale_feature_dataframe(df_clean):
    df_features = df_clean.drop(columns=["label"])
    train_cutoff = int(len(df_features) * TRAIN_RATIO)

    scaler = StandardScaler()
    scaler.fit(df_features.iloc[:train_cutoff])

    scaled_data = scaler.transform(df_features)
    print(f"Data scaled. Scaler fit on first {train_cutoff} rows strictly.")
    return scaled_data, scaler


def build_windows(scaled_data):
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled_data,
        window_shape=(WINDOW_SIZE, scaled_data.shape[1]),
    )
    windows = windows.squeeze(axis=1)
    print(f"Windowing complete. Tensor shape: {windows.shape}")
    return windows


def split_windows(windows):
    n_windows = len(windows)
    val_start = int(n_windows * TRAIN_RATIO)
    calib_start = val_start + int(n_windows * VAL_RATIO)
    test_start = calib_start + int(n_windows * CALIB_RATIO)

    splits = {
        "train": windows[:val_start],
        "val": windows[val_start:calib_start],
        "calib": windows[calib_start:test_start],
        "test": windows[test_start:],
    }

    print(f"X_train shape: {splits['train'].shape}")
    print(f"X_val shape:   {splits['val'].shape}")
    print(f"X_calib shape: {splits['calib'].shape}")
    print(f"X_test shape:  {splits['test'].shape}")
    return splits


def save_split_arrays(output_dir, splits):
    np.save(f"{output_dir}/X_train.npy", splits["train"])
    np.save(f"{output_dir}/X_val.npy", splits["val"])
    np.save(f"{output_dir}/X_calib.npy", splits["calib"])
    np.save(f"{output_dir}/X_test.npy", splits["test"])


def build_data_module(splits):
    data_module = AtacamaDataModule(
        x_train=splits["train"],
        x_val=splits["val"],
        x_calib=splits["calib"],
        x_test=splits["test"],
        batch_size=DATA_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    data_module.setup()
    sample_batch = next(iter(data_module.train_dataloader()))
    print(f"Batch shape from train_dataloader: {sample_batch.shape}")
    return data_module


def build_model(model_name, model_config):
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(
            f"Unknown model_name={model_name!r}. Available: {tuple(MODEL_REGISTRY)}"
        )
    model = model_cls(**model_config)
    print(model)
    return model


def build_trainer(output_dir):
    logger = CSVLogger(save_dir=output_dir, name="lstm_ae_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=TRAINER_CONFIG["max_epochs"],
        accelerator="auto",
        devices=TRAINER_CONFIG["devices"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=TRAINER_CONFIG["log_every_n_steps"],
    )
    return trainer, checkpoint_callback


def train_model(model, data_module, output_dir):
    trainer, checkpoint_callback = build_trainer(output_dir)
    print(f"Starting training. Logs and checkpoints will be saved to: {output_dir}")
    trainer.fit(model, datamodule=data_module)
    print("-" * 50)
    print("Training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path


def load_best_model(model_name, checkpoint_path, device):
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def run_model_inference(model, dataloader, device):
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def calculate_errors(true_data, pred_data):
    squared_error = np.square(true_data - pred_data)
    feature_mse = np.mean(squared_error, axis=1)
    window_mse = np.mean(squared_error, axis=(1, 2))
    return feature_mse, window_mse


def evaluate_reconstruction_model(model, data_module, splits, device):
    x_calib_hat = run_model_inference(model, data_module.predict_dataloader(), device)
    x_test_hat = run_model_inference(model, data_module.test_dataloader(), device)

    calib_feature_mse, calib_window_mse = calculate_errors(splits["calib"], x_calib_hat)
    test_feature_mse, test_window_mse = calculate_errors(splits["test"], x_test_hat)

    window_threshold = np.percentile(calib_window_mse, THRESHOLD_PERCENTILE)
    feature_thresholds = np.percentile(
        calib_feature_mse,
        THRESHOLD_PERCENTILE,
        axis=0,
    )

    print(f"Calibration Arrays Shape: {splits['calib'].shape}")
    print(f"Test Arrays Shape: {splits['test'].shape}")
    print(
        f"Global Window Threshold ({THRESHOLD_PERCENTILE}th percentile): "
        f"{window_threshold:.4f}"
    )

    return {
        "x_calib_hat": x_calib_hat,
        "x_test_hat": x_test_hat,
        "calib_feature_mse": calib_feature_mse,
        "calib_window_mse": calib_window_mse,
        "test_feature_mse": test_feature_mse,
        "test_window_mse": test_window_mse,
        "window_threshold": window_threshold,
        "feature_thresholds": feature_thresholds,
    }


def save_evaluation_outputs(output_dir, evaluation, feature_names):
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    test_window_anomalies = (
        evaluation["test_window_mse"] > evaluation["window_threshold"]
    ).astype(int)
    test_feature_anomalies = (
        evaluation["test_feature_mse"] > evaluation["feature_thresholds"]
    ).astype(int)

    np.save(os.path.join(eval_dir, "window_threshold.npy"), evaluation["window_threshold"])
    np.save(os.path.join(eval_dir, "feature_thresholds.npy"), evaluation["feature_thresholds"])
    np.save(os.path.join(eval_dir, "test_window_mse.npy"), evaluation["test_window_mse"])
    np.save(os.path.join(eval_dir, "test_feature_mse.npy"), evaluation["test_feature_mse"])
    np.save(os.path.join(eval_dir, "test_window_anomalies.npy"), test_window_anomalies)
    np.save(os.path.join(eval_dir, "test_feature_anomalies.npy"), test_feature_anomalies)

    print("Feature-Level Thresholds:")
    for feat, thresh in zip(feature_names, evaluation["feature_thresholds"]):
        print(f" - {feat}: {thresh:.4f}")
    print(f"Total Window Anomalies Found in Test Set: {test_window_anomalies.sum()}")
    print(f"All evaluation outputs and anomaly scores successfully stored in: {eval_dir}")

    return eval_dir, test_window_anomalies


def _safe_log_value(value):
    if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
        return value
    return str(value)


def build_cftsad_explainers(model, normal_core, threshold):
    base_kwargs = {
        "model": model,
        "normal_core": normal_core,
        "threshold": float(threshold),
        **CFTSAD_BASE_CONFIG,
    }

    explainers = {}
    for method, overrides in CFTSAD_METHOD_CONFIGS.items():
        explainers[method] = CounterfactualExplainer(
            method=method,
            **base_kwargs,
            **overrides,
        )
    return explainers


def run_counterfactual_benchmark(
    explainers,
    x_anomaly,
    idx_to_explain,
    original_score,
    threshold,
    eval_dir,
):
    csv_path = os.path.join(eval_dir, "counterfactual_log.csv")
    cf_arrays_dir = os.path.join(eval_dir, "cf_arrays")
    os.makedirs(cf_arrays_dir, exist_ok=True)

    rows = []
    results_by_method = {}
    for method_name, explainer in explainers.items():
        result = explainer.explain(x_anomaly)
        results_by_method[method_name] = result

        row = {
            "method": method_name,
            "test_index": int(idx_to_explain),
            "original_score": float(original_score),
            "target_threshold": float(threshold),
            "status": "success" if isinstance(result, CFResult) else "failed",
        }

        if isinstance(result, CFResult):
            print(f"[{method_name}] success -> cf_score={result.score_cf:.4f}")
            row["cf_score"] = float(result.score_cf)
            row["reason"] = "N/A"
            row["message"] = "N/A"
            cf_filename = f"cf_{method_name}_window_{idx_to_explain}.npy"
            np.save(os.path.join(cf_arrays_dir, cf_filename), result.x_cf)
            row["cf_array_file"] = cf_filename
            for key, value in result.meta.items():
                row[f"meta_{key}"] = _safe_log_value(value)
        else:
            print(f"[{method_name}] failed -> {result.reason}")
            row["cf_score"] = np.nan
            row["reason"] = result.reason
            row["message"] = result.message
            row["cf_array_file"] = "N/A"
            for key, value in result.diagnostics.items():
                row[f"diag_{key}"] = _safe_log_value(value)

        rows.append(row)

    df_log = pd.DataFrame(rows)
    if not os.path.exists(csv_path):
        df_log.to_csv(csv_path, index=False)
    else:
        df_log.to_csv(csv_path, mode="a", header=False, index=False)

    print(f"Counterfactual results appended to: {csv_path}")
    return results_by_method


def run_counterfactual_pipeline(model, splits, evaluation, eval_dir):
    anomaly_indices = np.where(
        evaluation["test_window_mse"] > evaluation["window_threshold"]
    )[0]
    if len(anomaly_indices) == 0:
        raise RuntimeError("No anomalies found in test set for counterfactual generation.")

    explainers = build_cftsad_explainers(
        model=model,
        normal_core=splits["calib"],
        threshold=evaluation["window_threshold"],
    )

    print(f"Found {len(anomaly_indices)} anomaly windows in test set.")
    print(f"Target threshold: {evaluation['window_threshold']:.4f}")
    print("Explainers initialized:", ", ".join(explainers.keys()))

    results_by_index = {}
    for idx in anomaly_indices:
        idx_to_explain = int(idx)
        x_anomaly = splits["test"][idx_to_explain]
        original_score = float(evaluation["test_window_mse"][idx_to_explain])

        print(f"\nGenerating counterfactuals for Test Window Index: {idx_to_explain}")
        print(f"Original score: {original_score:.4f}")

        results_by_method = run_counterfactual_benchmark(
            explainers=explainers,
            x_anomaly=x_anomaly,
            idx_to_explain=idx_to_explain,
            original_score=original_score,
            threshold=evaluation["window_threshold"],
            eval_dir=eval_dir,
        )
        results_by_index[idx_to_explain] = results_by_method

    return results_by_index


def main():
    device = configure_runtime()
    output_dir = make_output_dir()
    print(f"Results for this session will be stored in: {output_dir}")

    feature_names = load_feature_names(VAR_LIST_PATH)
    df_clean = load_and_clean_dataframe(DATA_PATH, feature_names, output_dir)
    scaled_data, _ = scale_feature_dataframe(df_clean)
    windows = build_windows(scaled_data)
    splits = split_windows(windows)
    save_split_arrays(output_dir, splits)

    data_module = build_data_module(splits)
    model = build_model(MODEL_NAME, MODEL_CONFIG)
    best_model_path = train_model(model, data_module, output_dir)
    best_model = load_best_model(MODEL_NAME, best_model_path, device)

    evaluation = evaluate_reconstruction_model(best_model, data_module, splits, device)
    eval_dir, _ = save_evaluation_outputs(output_dir, evaluation, feature_names)
    results_by_index = run_counterfactual_pipeline(
        best_model,
        splits,
        evaluation,
        eval_dir,
    )

    return results_by_index


if __name__ == "__main__":
    results_by_index = main()
    print(results_by_index)
