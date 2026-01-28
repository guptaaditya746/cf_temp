from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .io import ensure_dir, save_json


@dataclass
class TraceLogger:
    path: Path

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        ensure_dir(self.path.parent)
        rec = {"ts": time.time(), "event": event, **payload}
        # JSONL append
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(__import__("json").dumps(rec) + "\n")


def write_summary(path: str | Path, summary: Dict[str, Any]) -> None:
    save_json(path, summary)
