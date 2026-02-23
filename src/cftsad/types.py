from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class CFResult:
    x_cf: np.ndarray
    score_cf: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CFFailure:
    reason: str
    message: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
