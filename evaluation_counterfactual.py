import argparse

from experiments.stages import run_counterfactual_evaluation_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    summary = run_counterfactual_evaluation_stage(args.run_dir)
    print(summary)


if __name__ == "__main__":
    main()
