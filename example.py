#!/usr/bin/env python
# coding: utf-8

from experiments.runtime import configure_runtime
from experiments.stages import (
    run_counterfactual_evaluation_stage,
    run_counterfactual_generation_stage,
    run_model_evaluation_stage,
    run_preprocessing,
    run_training_stage,
)


def main():
    configure_runtime()
    run_dir = run_preprocessing()
    run_training_stage(run_dir)
    run_model_evaluation_stage(run_dir)
    run_counterfactual_generation_stage(run_dir)
    summary = run_counterfactual_evaluation_stage(run_dir)
    print(f"End-to-end run directory: {run_dir}")
    return summary


if __name__ == "__main__":
    print(main())
