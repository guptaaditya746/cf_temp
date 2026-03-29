import argparse

from experiments.stages import run_counterfactual_generation_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    results = run_counterfactual_generation_stage(args.run_dir)
    print(results)


if __name__ == "__main__":
    main()
