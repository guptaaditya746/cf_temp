import argparse

from experiments.stages import run_preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = run_preprocessing(args.run_dir)
    print(f"Preprocessing complete in: {run_dir}")


if __name__ == "__main__":
    main()
