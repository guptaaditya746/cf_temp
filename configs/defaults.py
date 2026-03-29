DATA_PATH = "./artifacts/Atacama.pkl"
VAR_LIST_PATH = "./artifacts/var_list.csv"

WINDOW_SIZE = 24
WINDOW_STRIDE = 1
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
    "devices": "auto",
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
    # "genetic": {
    #     "population_size": 50,
    #     "n_generations": 20,
    #     "use_plausibility_objective": True,
    #     "structured_mutation_weight": 0.35,
    #     "top_m_solutions": 5,
    #     "early_stop_patience": 15,
    #     "fallback_methods": ("segment", "motif", "nearest"),
    # },
}

ANOMALY_REPAIR_CONFIG = {
    "td": 2,
    "n_samples": 25,
    "interval_quantile": 0.9,
    "min_interval_length": 2,
    "enforce_psd": True,
    "random_seed": 42,
}
