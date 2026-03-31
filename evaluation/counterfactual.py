import os

import pandas as pd


def summarize_counterfactual_log(eval_dir):
    csv_path = os.path.join(eval_dir, "counterfactual_log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing counterfactual log: {csv_path}")

    df = pd.read_csv(csv_path)
    summary = (
        df.groupby("method", dropna=False)["status"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary_path = os.path.join(eval_dir, "counterfactual_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Counterfactual summary written to: {summary_path}")
    return summary
