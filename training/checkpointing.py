"""Checkpoint save/load helpers with optional Google Drive sync."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from utils.logging_utils import log


def _unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def _save(path: str, model, optimizer, scaler, step: int, val_loss: float | None, config: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "val_loss": val_loss,
        "config": config,
        "timestamp": datetime.utcnow().isoformat(),
    }
    torch.save(checkpoint, out_path)
    log(f"Saved checkpoint: {out_path}")


def _sync_to_gdrive(local_path: str) -> None:
    gdrive_path = os.environ.get("GDRIVE_PATH")
    if not gdrive_path:
        return
    try:
        Path(gdrive_path).mkdir(parents=True, exist_ok=True)
        dest = os.path.join(gdrive_path, os.path.basename(local_path))
        shutil.copy2(local_path, dest)
        log(f"Synced {local_path} -> {dest}")
    except Exception as exc:
        log(f"GDrive sync failed (non-fatal): {exc}")


def save_best(model, optimizer, scaler, step: int, val_loss: float | None, config: Any) -> None:
    """Save to checkpoints/best.pt."""
    path = "checkpoints/best.pt"
    _save(path, model, optimizer, scaler, step, val_loss, config)
    _sync_to_gdrive(path)


def save_latest(model, optimizer, scaler, step: int, val_loss: float | None, config: Any) -> None:
    """Save to checkpoints/latest.pt."""
    path = "checkpoints/latest.pt"
    _save(path, model, optimizer, scaler, step, val_loss, config)
    _sync_to_gdrive(path)


def load_checkpoint(path: str, model, optimizer=None, scaler=None, device: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    log(f"Loaded checkpoint {path} at step {checkpoint.get('step', 0)}")
    return checkpoint
