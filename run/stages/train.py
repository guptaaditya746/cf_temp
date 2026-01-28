from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.io import load_npz, save_yaml
from ..utils.model_io import export_torchscript


class Conv1dAE(nn.Module):
    """
    Simple reconstruction model for (N,L,F) windows.
    Uses convs on time with channels=F.
    """

    def __init__(
        self,
        F: int,
        channels: list[int],
        latent: int,
        kernel: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        k = int(kernel)
        pad = k // 2

        enc = []
        c_in = F
        for c in channels:
            enc += [nn.Conv1d(c_in, c, kernel_size=k, padding=pad), nn.ReLU()]
            if dropout and dropout > 0:
                enc += [nn.Dropout(p=float(dropout))]
            c_in = c
        enc += [nn.Conv1d(c_in, latent, kernel_size=1), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)

        dec = []
        c_in = latent
        for c in reversed(channels):
            dec += [nn.Conv1d(c_in, c, kernel_size=k, padding=pad), nn.ReLU()]
            if dropout and dropout > 0:
                dec += [nn.Dropout(p=float(dropout))]
            c_in = c
        dec += [nn.Conv1d(c_in, F, kernel_size=1)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,L,F) -> (N,F,L)
        x_ch = x.transpose(1, 2)
        z = self.encoder(x_ch)
        y = self.decoder(z)
        # back to (N,L,F)
        return y.transpose(1, 2)


@dataclass(frozen=True)
class TrainOut:
    model_ts_path: Path
    model_state_path: Path
    config_model_path: Path


def _device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_train(run_dir: Path, cfg: Dict[str, Any]) -> TrainOut:
    art = run_dir / "artifacts"

    train = load_npz(art / "train.npz")["x"]
    valid = load_npz(art / "validation.npz")["x"]

    if train.ndim != 3:
        raise ValueError(f"train.x must be (N,L,F). Got {train.shape}")
    N, L, F = train.shape

    mcfg = cfg["model"]
    dev = _device(str(mcfg.get("device", "cpu")))
    model = Conv1dAE(
        F=F,
        channels=list(mcfg["channels"]),
        latent=int(mcfg["latent"]),
        kernel=int(mcfg["kernel"]),
        dropout=float(mcfg.get("dropout", 0.0)),
    ).to(dev)

    lr = float(mcfg["lr"])
    bs = int(mcfg["batch_size"])
    epochs = int(mcfg["epochs"])

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loss: MSE of reconstruction (this is training, not scoring; allowed)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    train_ds = TensorDataset(torch.from_numpy(train))
    valid_ds = TensorDataset(torch.from_numpy(valid))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for (xb,) in train_dl:
            xb = xb.to(dev)
            opt.zero_grad(set_to_none=True)
            xh = model(xb)
            loss = loss_fn(xh, xb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.shape[0]
            n_tr += xb.shape[0]
        tr_loss /= max(1, n_tr)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for (xb,) in valid_dl:
                xb = xb.to(dev)
                xh = model(xb)
                loss = loss_fn(xh, xb)
                va_loss += float(loss.item()) * xb.shape[0]
                n_va += xb.shape[0]
        va_loss /= max(1, n_va)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"[train] epoch={ep:03d} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} best_val={best_val:.6f}"
        )

    model_state_path = art / "model_state.pt"
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, model_state_path)

    # Load best state before exporting TorchScript
    model.load_state_dict(best_state)
    example = torch.from_numpy(train[:1]).to(dev)  # (1,L,F)
    model_ts_path = art / "model_ts.pt"
    export_torchscript(model, example, model_ts_path)

    config_model_path = art / "config_model.yaml"
    save_yaml(
        config_model_path,
        {
            "type": mcfg["type"],
            "input_shape": [None, int(L), int(F)],
            "channels": list(mcfg["channels"]),
            "latent": int(mcfg["latent"]),
            "kernel": int(mcfg["kernel"]),
            "dropout": float(mcfg.get("dropout", 0.0)),
            "training": {
                "lr": lr,
                "batch_size": bs,
                "epochs": epochs,
                "best_val": float(best_val),
                "device": dev,
            },
        },
    )

    return TrainOut(
        model_ts_path=model_ts_path,
        model_state_path=model_state_path,
        config_model_path=config_model_path,
    )
