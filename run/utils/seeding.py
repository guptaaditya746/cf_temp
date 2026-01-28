from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:
    torch = None


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic_torch: bool = True


def seed_everything(cfg: SeedConfig) -> None:
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cfg.deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
