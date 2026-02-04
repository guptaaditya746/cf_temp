# utils/exp_logger.py
import json
import time
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def utcstamp():
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def to_jsonable(x: Any):
    # Minimal + safe conversion (won't explode on tensors/arrays)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach()
            return {
                "_type": "torch.Tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "min": float(x.min().item()) if x.numel() else None,
                "max": float(x.max().item()) if x.numel() else None,
                "mean": float(x.float().mean().item()) if x.numel() else None,
            }
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        return {
            "_type": "np.ndarray",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "min": float(np.nanmin(x)) if x.size else None,
            "max": float(np.nanmax(x)) if x.size else None,
            "mean": float(np.nanmean(x)) if x.size else None,
        }

    if is_dataclass(x):
        return asdict(x)

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return str(x)


class ExperimentLogger:
    def __init__(self, out_dir: str, tag: str = "xai_stageB"):
        self.run_id = f"{utcstamp()}__{tag}__run-{uuid.uuid4().hex[:6]}"
        self.root = Path(out_dir) / "runs" / self.run_id
        ensure_dir(self.root)
        self.events_path = self.root / "events.jsonl"
        self._t0 = time.time()

        self.write_json(
            self.root / "run_meta.json",
            {
                "run_id": self.run_id,
                "created_utc": utcstamp(),
                "tag": tag,
            },
        )

    def log_event(self, event: Dict[str, Any]):
        event = dict(event)
        event["ts"] = time.time()
        event["t_rel_sec"] = event["ts"] - self._t0
        with open(self.events_path, "a") as f:
            f.write(json.dumps(to_jsonable(event)) + "\n")

    def method_dir(self, method: str, anomaly_idx: int) -> Path:
        p = self.root / "methods" / method / f"idx{int(anomaly_idx)}"
        ensure_dir(p)
        return p

    def write_json(self, path: str | Path, obj: Any):
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "w") as f:
            json.dump(to_jsonable(obj), f, indent=2)

    def save_npz(self, path: str | Path, **arrays):
        path = Path(path)
        ensure_dir(path.parent)
        np.savez_compressed(path, **arrays)

    def save_fig(self, path: str | Path):
        path = Path(path)
        ensure_dir(path.parent)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    def capture_exception(self, path: str | Path, e: Exception):
        self.write_json(
            path,
            {
                "exception": repr(e),
                "traceback": traceback.format_exc(),
            },
        )
        self.write_json(path, {
            "exception": repr(e),
            "traceback": traceback.format_exc(),
        })
