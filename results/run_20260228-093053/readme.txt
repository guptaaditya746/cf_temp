reduce to belwo 

base_kwargs = {
    "model": model_predict_fn,
    "normal_core": normal_core,
    "threshold": float(threshold),
    "use_constraints_v2": True,
    "enable_fallback_chain": True,
    "fallback_retry_budget": 2,
    "normal_core_threshold_quantile": 0.95,
    "normal_core_filter_factor": 1.0,
    "normal_core_max_size": 100,  # <- reduce core to at most 500 windows
    "normal_core_use_diversity_sampling": True,
    "random_seed": 42,
}
