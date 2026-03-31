import argparse

from experiments.stages import run_training_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    best_model_path = run_training_stage(args.run_dir)
    print(f"Training complete. Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
