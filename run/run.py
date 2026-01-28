from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from stages.calibrate import run_calibrate
from stages.counterfactuals import run_counterfactuals
from stages.infer import run_infer
from stages.preprocess import run_preprocess
from stages.select_targets import run_select_targets
from stages.train import run_train
from utils.config import (
    freeze_config_to_run,
    get_default_config,
    load_config_file,
    resolve_run_dir,
)
from utils.io import ensure_dir, save_json
from utils.seeding import SeedConfig, seed_everything


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("run pipeline")
    p.add_argument(
        "--dataset", required=True, type=str, help="dataset name (e.g., atacama)"
    )
    p.add_argument("--run_name", required=True, type=str, help="run name (e.g., exp01)")
    p.add_argument(
        "--config", default=None, type=str, help="optional YAML config override"
    )
    p.add_argument(
        "--base_dir", default=None, type=str, help="override base_dir in config"
    )
    p.add_argument("--seed", default=None, type=int, help="override seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = get_default_config()
    if args.config is not None:
        cfg_file = load_config_file(args.config)
        cfg = _deep_update(cfg, cfg_file)

    if args.base_dir is not None:
        cfg["base_dir"] = args.base_dir
    if args.seed is not None:
        cfg["seed"]["seed"] = int(args.seed)

    paths = resolve_run_dir(cfg["base_dir"], args.dataset, args.run_name)

    # Freeze resolved config into run directory (reproducibility)
    freeze_config_to_run(cfg, paths.artifacts / "config_frozen.yaml")

    # Deterministic seeding
    seed_everything(SeedConfig(**cfg["seed"]))

    produced = {}

    # 1) preprocess
    out_pre = run_preprocess(paths.root, args.dataset, cfg)
    produced["preprocess"] = {k: str(v) for k, v in out_pre.__dict__.items()}

    # 2) train
    out_tr = run_train(paths.root, cfg)
    produced["train"] = {k: str(v) for k, v in out_tr.__dict__.items()}

    # 3) calibrate
    out_cal = run_calibrate(paths.root, cfg)
    produced["calibrate"] = {k: str(v) for k, v in out_cal.__dict__.items()}

    # 4) infer
    out_inf = run_infer(paths.root, cfg)
    produced["infer"] = {k: str(v) for k, v in out_inf.__dict__.items()}

    # 5) select targets
    out_sel = run_select_targets(paths.root, cfg)
    produced["select_targets"] = {k: str(v) for k, v in out_sel.__dict__.items()}

    # 6) counterfactuals
    out_cf = run_counterfactuals(paths.root, cfg)
    produced["counterfactuals"] = {k: str(v) for k, v in out_cf.__dict__.items()}

    # Summary
    save_json(paths.root / "run_summary.json", produced)

    print("\n=== RUN COMPLETE ===")
    print(f"run_dir: {paths.root}")
    print("Artifacts written:")
    for stage, items in produced.items():
        print(f"  [{stage}]")
        for _, p in items.items():
            print(f"    - {p}")


if __name__ == "__main__":
    main()
