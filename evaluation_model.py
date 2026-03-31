import argparse

from experiments.stages import run_model_evaluation_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    eval_dir, anomaly_count = run_model_evaluation_stage(args.run_dir)
    print(f"Model evaluation saved to: {eval_dir}")
    print(f"Detected anomalous windows: {anomaly_count}")


if __name__ == "__main__":
    main()
