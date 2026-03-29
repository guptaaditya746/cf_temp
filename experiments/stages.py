import os

import numpy as np

from configs.defaults import DATA_PATH, MODEL_CONFIG, MODEL_NAME, VAR_LIST_PATH
from experiments.runtime import configure_runtime, make_output_dir
from utils.storage import ensure_dir, load_json, save_json


def _load_splits(run_dir):
    return {
        "train": np.load(os.path.join(run_dir, "X_train.npy")),
        "val": np.load(os.path.join(run_dir, "X_val.npy")),
        "calib": np.load(os.path.join(run_dir, "X_calib.npy")),
        "test": np.load(os.path.join(run_dir, "X_test.npy")),
    }


def run_preprocessing(run_dir=None):
    from preprocessing.data import (
        build_windows,
        load_and_clean_dataframe,
        load_feature_names,
        save_split_arrays,
        scale_feature_dataframe,
        split_windows,
    )

    if run_dir is None:
        run_dir = make_output_dir()
    ensure_dir(run_dir)

    feature_names = load_feature_names(VAR_LIST_PATH)
    df_clean = load_and_clean_dataframe(DATA_PATH, feature_names, run_dir)
    scaled_data, _ = scale_feature_dataframe(df_clean)
    windows = build_windows(scaled_data)
    splits = split_windows(windows)
    save_split_arrays(run_dir, splits)

    save_json(
        os.path.join(run_dir, "preprocessing_metadata.json"),
        {
            "feature_names": feature_names,
            "data_path": DATA_PATH,
            "var_list_path": VAR_LIST_PATH,
        },
    )
    return run_dir


def run_training_stage(run_dir):
    from experiments.training import build_data_module, train_model
    from models.autoencoder import build_model

    splits = _load_splits(run_dir)
    data_module = build_data_module(splits)
    sample_batch = next(iter(data_module.train_dataloader()))
    print(f"Batch shape from train_dataloader: {sample_batch.shape}")

    model = build_model(MODEL_NAME, MODEL_CONFIG)
    print(model)
    best_model_path = train_model(model, data_module, run_dir)

    save_json(
        os.path.join(run_dir, "training_metadata.json"),
        {
            "model_name": MODEL_NAME,
            "best_model_path": best_model_path,
        },
    )
    return best_model_path


def run_model_evaluation_stage(run_dir):
    from evaluation.reconstruction import evaluate_reconstruction_model, save_evaluation_outputs
    from experiments.training import build_data_module
    from models.autoencoder import load_model_from_checkpoint

    device = configure_runtime()
    splits = _load_splits(run_dir)
    data_module = build_data_module(splits)
    training_metadata = load_json(os.path.join(run_dir, "training_metadata.json"))
    preprocessing_metadata = load_json(os.path.join(run_dir, "preprocessing_metadata.json"))

    model = load_model_from_checkpoint(
        training_metadata["model_name"],
        training_metadata["best_model_path"],
        device,
    )
    evaluation = evaluate_reconstruction_model(model, data_module, splits, device)
    eval_dir, test_window_anomalies = save_evaluation_outputs(
        run_dir,
        evaluation,
        preprocessing_metadata["feature_names"],
    )
    return eval_dir, int(test_window_anomalies.sum())


def run_counterfactual_generation_stage(run_dir):
    from counterfactual_methods.pipeline import run_counterfactual_pipeline
    from models.autoencoder import load_model_from_checkpoint

    device = configure_runtime()
    splits = _load_splits(run_dir)
    training_metadata = load_json(os.path.join(run_dir, "training_metadata.json"))
    model = load_model_from_checkpoint(
        training_metadata["model_name"],
        training_metadata["best_model_path"],
        device,
    )

    evaluation = {
        "test_window_mse": np.load(os.path.join(run_dir, "evaluation", "test_window_mse.npy")),
        "window_threshold": float(
            np.load(os.path.join(run_dir, "evaluation", "window_threshold.npy"))
        ),
    }
    eval_dir = os.path.join(run_dir, "evaluation")
    return run_counterfactual_pipeline(model, splits, evaluation, eval_dir)


def run_counterfactual_evaluation_stage(run_dir):
    from evaluation.counterfactual import summarize_counterfactual_log

    eval_dir = os.path.join(run_dir, "evaluation")
    return summarize_counterfactual_log(eval_dir)
