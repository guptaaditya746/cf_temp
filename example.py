#!/usr/bin/env python
# coding: utf-8

# ## end-to-end pipeline in a way that matches the Darban et al. reconstruction-based TSAD framing



# In[1]:




# Goal: Train an unsupervised reconstruction model on normal behavior and detect collective (subsequence) anomalies as contiguous events in a multivariate time series. The system produces timestamp-level anomaly scores (for thresholding)

# In[2]:


import cftsad
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os



import random
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning Framework (Pick your flavor)
import torch
import torch.nn as nn


# For PyTorch
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

# For TensorFlow


print(f"Currently using: {device}")

# Ensure plots show up inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' # High-res plots")

# Ignore annoying warnings (e.g., DeprecationWarnings)
import warnings
warnings.filterwarnings('ignore')

# Visual style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")




# In[ ]:





# In[ ]:





# In[3]:


from datetime import datetime

# Create a unique run ID for saving results
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f"results/run_{RUN_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Results for this session will be stored in: {OUTPUT_DIR}")


import pandas as pd
import numpy as np
import os

# --- 1. Ingest ---
data_path = '../cf/artifacts/Atacama.pkl'
var_list_path = '../cf/artifacts/var_list.csv'

# Load expected features
with open(var_list_path, 'r') as f:
    FEATURES = [line.strip() for line in f.readlines() if line.strip()]

df = pd.read_pickle(data_path)

# --- 2. Timestamp Alignment ---
# Convert UNIX timestamp float to datetime and set as index
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.set_index('time')
df = df.sort_index()

# --- 3. Quality Assurance (QA) ---
# Remove duplicates
df = df[~df.index.duplicated(keep='first')]

# Enforce consistent hourly sampling frequency
df = df.asfreq('H')

# Resolve missing values (Forward fill short gaps to prevent leakage, median for the rest)
df[FEATURES] = df[FEATURES].ffill(limit=3)
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

# Handle labels (Binarize anomalies based on Readme: > 2.0mm rain)
df['label'] = df['label'].fillna(0).astype(int)
df.loc[df['precipitation (mm)'] > 2.0, 'label'] = 1

# --- 4. Freeze Schema ---
# Lock the ordering: Features first, then label.
# This exact order will be reused for all splits.
FINAL_COLUMNS = FEATURES + ['label']
df_clean = df[FINAL_COLUMNS]

# Save this checkpoint to your tracking directory
df_clean.to_pickle(f"{OUTPUT_DIR}/01_ingested_qa_data.pkl")

print(f"Cleaned Shape: {df_clean.shape}")
print(f"Schema Locked. Features: {FEATURES}")
df_clean.head()


# In[5]:


from sklearn.preprocessing import StandardScaler
import joblib

# 1. Drop the label column completely
df_features = df_clean.drop(columns=['label'])

# 2. Determine the training cutoff index for fitting the scaler
total_len = len(df_features)
train_cutoff = int(total_len * 0.6)

# 3. Fit scaler ONLY on the training slice
scaler = StandardScaler()
scaler.fit(df_features.iloc[:train_cutoff])

# 4. Transform the entire dataset at once
scaled_data = scaler.transform(df_features)

# Save the scaler to your tracking directory so evaluation can use it later
# joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

print(f"Data scaled. Scaler fit on first {train_cutoff} rows strictly.")


# In[6]:


import numpy as np

window_size = 24  # Lookback of 24 hours
stride = 1        # Slide forward by 1 hour at a time

# Create rolling windows over the 2D scaled data
# Output shape: (num_windows, features, window_size)
windows = np.lib.stride_tricks.sliding_window_view(
    scaled_data,
    window_shape=(window_size, scaled_data.shape[1])
)

# Squeeze and transpose to get the standard PyTorch shape: (Batch, Sequence_Length, Features)
windows = windows.squeeze(axis=1)

print(f"Windowing complete. Tensor shape: {windows.shape}")


# In[7]:


n_windows = len(windows)

# Calculate split indices based on the windowed array length
# 60% Train, 10% Val, 10% Calib, 20% Test
val_start = int(n_windows * 0.6)
calib_start = val_start + int(n_windows * 0.1)
test_start = calib_start + int(n_windows * 0.1)

# Slice the arrays
X_train = windows[:val_start]
X_val = windows[val_start:calib_start]
X_calib = windows[calib_start:test_start]
X_test = windows[test_start:]

# Sanity check the shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"X_calib shape: {X_calib.shape}")
print(f"X_test shape:  {X_test.shape}")

# Save the windowed splits to your results directory
np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
np.save(f"{OUTPUT_DIR}/X_calib.npy", X_calib)
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)


# In[8]:


import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class UnsupervisedTimeSeriesDataset(Dataset):
    """
    A simple dataset that only yields the input feature windows.
    """
    def __init__(self, data_array):
        # Convert the numpy array to a PyTorch float32 tensor
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Strictly returning only the feature window: shape (window_size, num_features)
        return self.data[idx]


# In[9]:


class AtacamaDataModule(pl.LightningDataModule):
    def __init__(self, X_train, X_val, X_calib, X_test, batch_size=64, num_workers=2):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.X_calib = X_calib
        self.X_test = X_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = UnsupervisedTimeSeriesDataset(self.X_train)
            self.val_dataset = UnsupervisedTimeSeriesDataset(self.X_val)

        # Assign test dataset for use in dataloader
        if stage == 'test' or stage is None:
            self.test_dataset = UnsupervisedTimeSeriesDataset(self.X_test)

        # Assign calibration dataset (can be used in predict or a custom stage)
        if stage == 'predict' or stage is None:
            self.calib_dataset = UnsupervisedTimeSeriesDataset(self.X_calib)

    def train_dataloader(self):
        # Shuffle is True ONLY for the training set
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        # We can map your 'calibration' split to the predict dataloader
        return DataLoader(
            self.calib_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# In[10]:


# Instantiate the DataModule
# We use the arrays (X_train, X_val, X_calib, X_test) created in the windowing step
data_module = AtacamaDataModule(
    X_train=X_train,
    X_val=X_val,
    X_calib=X_calib,
    X_test=X_test,
    batch_size=128,  # Adjust based on your GPU memory
    num_workers=8    # Usually set to number of CPU cores
)

# Call setup manually if you want to test the dataloaders before training
data_module.setup()

# Quick sanity check on the dataloader output
sample_batch = next(iter(data_module.train_dataloader()))
print(f"Batch shape from train_dataloader: {sample_batch.shape}")
# Expected output: torch.Size([128, 24, 6]) -> (Batch_Size, Window_Size, Num_Features)


# In[11]:


import torch
import torch.nn as nn
import pytorch_lightning as pl

class TimeSeriesAutoencoder(pl.LightningModule):
    def __init__(self, seq_len=24, n_features=6, hidden_dim=64, latent_dim=16, lr=1e-3):
        super().__init__()
        self.save_hyperparameters() # Logs these params automatically to TensorBoard/W&B

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr

        # --- ENCODER ---
        self.encoder_lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.encoder_linear = nn.Linear(self.hidden_dim, self.latent_dim)

        # --- DECODER ---
        # The decoder takes the latent vector, so we need to expand it back to seq_len
        self.decoder_lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.decoder_linear = nn.Linear(self.hidden_dim, self.n_features)

        # --- LOSS FUNCTION ---
        self.criterion = nn.MSELoss()

    def encode(self, x):
        # x shape: (Batch, Seq_Len, Features)
        _, (hidden, _) = self.encoder_lstm(x)
        # hidden shape: (1, Batch, Hidden_Dim) -> squeeze to (Batch, Hidden_Dim)
        hidden = hidden.squeeze(0)
        latent = self.encoder_linear(hidden)
        # latent shape: (Batch, Latent_Dim)
        return latent

    def decode(self, latent):
        # Repeat the latent vector for the length of the sequence
        # latent shape: (Batch, Latent_Dim) -> (Batch, Seq_Len, Latent_Dim)
        latent_repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)

        decoder_out, _ = self.decoder_lstm(latent_repeated)
        # decoder_out shape: (Batch, Seq_Len, Hidden_Dim)

        reconstruction = self.decoder_linear(decoder_out)
        # reconstruction shape: (Batch, Seq_Len, Features)
        return reconstruction

    def forward(self, x):
        # The full pipeline: encode -> decode
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def training_step(self, batch, batch_idx):
        # batch is our windowed tensor 'x'
        x = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# In[12]:


# Create the model instance
# seq_len = 24 (your window size), n_features = length of your frozen schema
model = TimeSeriesAutoencoder(
    seq_len=24,
    n_features=6,
    hidden_dim=32,   # Compressing 6 features down
    latent_dim=8,    # Final bottleneck size
    lr=1e-3
)

print(model)


# In[13]:


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import os

# 1. Route logs to the unified tracking folder
logger = CSVLogger(save_dir=OUTPUT_DIR, name="lstm_ae_logs")

# 2. Save only the best model based on validation loss
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(OUTPUT_DIR, "checkpoints"),
    filename="best-model-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)

# 3. Stop training if the model stops improving for 5 epochs
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=True,
    mode="min"
)


# In[14]:


# Initialize the Lightning Trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="auto",
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    enable_progress_bar=True,
    log_every_n_steps=10
)


# In[15]:


print(f"Starting training. Logs and checkpoints will be saved to: {OUTPUT_DIR}")

# Execute the training loop
trainer.fit(model, datamodule=data_module)

print("-" * 50)
print(f"Training complete.")
print(f"Best model saved at: {checkpoint_callback.best_model_path}")


# In[16]:


import torch
import numpy as np

def get_reconstructions(model, dataloader):
    """Passes data through the model and returns inputs and reconstructions."""
    model.eval()
    # Ensure model is on the correct device (GPU/MPS/CPU)
    device = next(model.parameters()).device

    all_inputs = []
    all_reconstructions = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction = model(batch)

            all_inputs.append(batch.cpu().numpy())
            all_reconstructions.append(reconstruction.cpu().numpy())

    return np.concatenate(all_inputs), np.concatenate(all_reconstructions)

# Load the best model from the checkpoint saved during training
best_model_path = checkpoint_callback.best_model_path
model = TimeSeriesAutoencoder.load_from_checkpoint(best_model_path)

# Extract for Calibration and Test sets
calib_loader = data_module.predict_dataloader() # We mapped Calib here earlier
test_loader = data_module.test_dataloader()

X_calib_true, X_calib_hat = get_reconstructions(model, calib_loader)
X_test_true, X_test_hat = get_reconstructions(model, test_loader)

print(f"Calibration Arrays Shape: {X_calib_true.shape}")
print(f"Test Arrays Shape: {X_test_true.shape}")


# In[17]:


def calculate_errors(true_data, pred_data):
    # 1. Raw Squared Error: (Batch, Time, Features)
    squared_error = np.square(true_data - pred_data)

    # 2. Feature-Level MSE: Average across the Time dimension (axis=1)
    # Output Shape: (Batch, Features)
    feature_mse = np.mean(squared_error, axis=1)

    # 3. Window-Level MSE: Average across both Time and Features (axis=(1,2))
    # Output Shape: (Batch,)
    window_mse = np.mean(squared_error, axis=(1, 2))

    return feature_mse, window_mse

# Calculate for Calibration
calib_feature_mse, calib_window_mse = calculate_errors(X_calib_true, X_calib_hat)

# Calculate for Test
test_feature_mse, test_window_mse = calculate_errors(X_test_true, X_test_hat)


# In[18]:


# Set the threshold using the 99th percentile of the Calibration set
PERCENTILE = 99

# 1. Window-Level Threshold (A single float value)
window_threshold = np.percentile(calib_window_mse, PERCENTILE)

# 2. Feature-Level Thresholds (An array of floats, one for each feature)
feature_thresholds = np.percentile(calib_feature_mse, PERCENTILE, axis=0)

print(f"Global Window Threshold (99th percentile): {window_threshold:.4f}")
print("Feature-Level Thresholds:")
for feat, thresh in zip(FEATURES, feature_thresholds):
    print(f" - {feat}: {thresh:.4f}")


# In[19]:


import os

# 1. Score the Test Set (Boolean to Integer: True->1, False->0)
test_window_anomalies = (test_window_mse > window_threshold).astype(int)
test_feature_anomalies = (test_feature_mse > feature_thresholds).astype(int)

print(f"Total Window Anomalies Found in Test Set: {test_window_anomalies.sum()}")

# 2. Save all evaluation outputs to the unified run directory
eval_dir = os.path.join(OUTPUT_DIR, "evaluation")
os.makedirs(eval_dir, exist_ok=True)

# Save Thresholds
np.save(os.path.join(eval_dir, "window_threshold.npy"), window_threshold)
np.save(os.path.join(eval_dir, "feature_thresholds.npy"), feature_thresholds)

# Save Test Scores
np.save(os.path.join(eval_dir, "test_window_mse.npy"), test_window_mse)
np.save(os.path.join(eval_dir, "test_feature_mse.npy"), test_feature_mse)

# Save Binary Anomaly Flags
np.save(os.path.join(eval_dir, "test_window_anomalies.npy"), test_window_anomalies)
np.save(os.path.join(eval_dir, "test_feature_anomalies.npy"), test_feature_anomalies)

print(f"All evaluation outputs and anomaly scores successfully stored in: {eval_dir}")





# In[22]:


import numpy as np

# Find the indices of all windows flagged as anomalies in the Test set
anomaly_indices = np.where(test_window_anomalies == 1)[0]


# In[23]:


import torch
import numpy as np

# Import cftsad API from installed package or local src fallback.
try:
    from cftsad import CFFailure, CFResult, CounterfactualExplainer
except Exception:
    from src.cftsad import CFFailure, CFResult, CounterfactualExplainer


class AutoencoderPredictorAdapter:
    """
    Thin model adapter so cftsad can call the AE with a stable numpy API.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, x_np):
        x_arr = np.asarray(x_np, dtype=np.float32)
        self.model.eval()
        model_device = next(self.model.parameters()).device

        with torch.no_grad():
            x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=model_device)
            added_batch_dim = False
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0)
                added_batch_dim = True

            reconstruction = self.model(x_tensor).detach().cpu().numpy()
            if added_batch_dim:
                reconstruction = reconstruction.squeeze(0)
            return reconstruction


def _safe_log_value(v):
    if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
        return v
    return str(v)


def build_cftsad_explainers(model_predict_fn, normal_core, threshold):
    base_kwargs = {
        "model": model_predict_fn,
        "normal_core": normal_core,
        "threshold": float(threshold),
        "use_constraints_v2": True,
        "enable_fallback_chain": True,
        "fallback_retry_budget": 2,
        "normal_core_threshold_quantile": 0.95,
        "normal_core_filter_factor": 1.0,
        "random_seed": 42,
    }

    method_overrides = {
        "nearest": {
            "nearest_top_k": 10,
            "nearest_alpha_steps": 11,
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
            "motif_top_k": 10,
            "motif_n_segments": 4,
            "motif_length_factors": (0.75, 1.0, 1.25),
            "motif_context_weight": 0.2,
            "motif_use_affine_fit": True,
            "fallback_methods": ("segment", "nearest", "genetic"),
        },
        "genetic": {
            "population_size": 100,
            "n_generations": 50,
            "use_plausibility_objective": True,
            "structured_mutation_weight": 0.35,
            "top_m_solutions": 5,
            "early_stop_patience": 15,
            "fallback_methods": ("segment", "motif", "nearest"),
        },
    }

    explainers = {}
    for method, overrides in method_overrides.items():
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
            for k, v in result.meta.items():
                row[f"meta_{k}"] = _safe_log_value(v)
        else:
            print(f"[{method_name}] failed -> {result.reason}")
            row["cf_score"] = np.nan
            row["reason"] = result.reason
            row["message"] = result.message
            row["cf_array_file"] = "N/A"
            for k, v in result.diagnostics.items():
                row[f"diag_{k}"] = _safe_log_value(v)

        rows.append(row)

    df_log = pd.DataFrame(rows)
    if not os.path.exists(csv_path):
        df_log.to_csv(csv_path, index=False)
    else:
        df_log.to_csv(csv_path, mode='a', header=False, index=False)

    print(f"Counterfactual results appended to: {csv_path}")
    return results_by_method


if len(anomaly_indices) == 0:
    raise RuntimeError("No anomalies found in test set for counterfactual generation.")

predict_fn = AutoencoderPredictorAdapter(model)
explainers = build_cftsad_explainers(
    model_predict_fn=predict_fn,
    normal_core=X_train,
    threshold=window_threshold,
)

print(f"Found {len(anomaly_indices)} anomaly windows in test set.")
print(f"Target threshold: {window_threshold:.4f}")
print("Explainers initialized:", ", ".join(explainers.keys()))

results_by_index = {}
for idx in anomaly_indices:
    idx_to_explain = int(idx)
    x_anomaly = X_test[idx_to_explain]
    original_score = float(test_window_mse[idx_to_explain])

    print(f"\nGenerating counterfactuals for Test Window Index: {idx_to_explain}")
    print(f"Original score: {original_score:.4f}")

    results_by_method = run_counterfactual_benchmark(
        explainers=explainers,
        x_anomaly=x_anomaly,
        idx_to_explain=idx_to_explain,
        original_score=original_score,
        threshold=window_threshold,
        eval_dir=eval_dir,
    )
    results_by_index[idx_to_explain] = results_by_method


# In[ ]:


results_by_index
